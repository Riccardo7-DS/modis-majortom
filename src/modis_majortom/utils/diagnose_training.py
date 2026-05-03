"""
Diagnostic script to identify why cloud segmentation metrics are low.
Run this to check data quality and class distribution.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def diagnose_cloud_labels(dataset, output_dir="diagnostics", num_samples=10):
    """
    Visualize training samples to check if cloud labels are correct.
    
    Args:
        dataset: MODISZarrPatchDataset instance
        output_dir: Directory to save diagnostic plots
        num_samples: Number of samples to visualize
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Visualizing {num_samples} samples to check label quality...")
    
    cloud_fractions = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            x, y = dataset[i]
            
            # Convert to numpy
            red = x[0].numpy()
            nir = x[1].numpy()
            ndvi = x[2].numpy()
            cloud_mask = y.numpy()
            
            cloud_fraction = cloud_mask.mean()
            cloud_fractions.append(cloud_fraction)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Input bands
            axes[0, 0].imshow(red, cmap='gray')
            axes[0, 0].set_title(f'Red Band\n(min: {red.min():.3f}, max: {red.max():.3f})')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(nir, cmap='gray')
            axes[0, 1].set_title(f'NIR Band\n(min: {nir.min():.3f}, max: {nir.max():.3f})')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
            axes[0, 2].set_title(f'NDVI\n(min: {ndvi.min():.3f}, max: {ndvi.max():.3f})')
            axes[0, 2].axis('off')
            
            # Row 2: Cloud mask analysis
            axes[1, 0].imshow(cloud_mask, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title(f'Cloud Mask (Ground Truth)\nCloud %: {cloud_fraction:.2%}')
            axes[1, 0].axis('off')
            
            # Overlay cloud mask on NDVI
            ndvi_rgb = plt.cm.RdYlGn((ndvi + 1) / 2)  # Normalize to [0, 1]
            cloud_overlay = ndvi_rgb.copy()
            cloud_overlay[cloud_mask > 0.5] = [1, 0, 0, 1]  # Red for cloud
            axes[1, 1].imshow(cloud_overlay)
            axes[1, 1].set_title('Cloud Overlay on NDVI\n(Red = Cloud)')
            axes[1, 1].axis('off')
            
            # Histogram of NDVI values
            axes[1, 2].hist(ndvi.flatten(), bins=50, edgecolor='black')
            axes[1, 2].set_title('NDVI Distribution')
            axes[1, 2].set_xlabel('NDVI')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].axvline(x=0, color='r', linestyle='--', label='NDVI=0')
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(output_path / f'sample_{i:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Sample {i}: Cloud fraction = {cloud_fraction:.4f} ({cloud_fraction:.2%})")
            
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            continue
    
    # Summary statistics
    if cloud_fractions:
        cloud_fractions = np.array(cloud_fractions)
        logger.info(f"\n{'='*60}")
        logger.info(f"Cloud Fraction Statistics:")
        logger.info(f"  Mean: {cloud_fractions.mean():.4f} ({cloud_fractions.mean():.2%})")
        logger.info(f"  Std:  {cloud_fractions.std():.4f}")
        logger.info(f"  Min:  {cloud_fractions.min():.4f} ({cloud_fractions.min():.2%})")
        logger.info(f"  Max:  {cloud_fractions.max():.4f} ({cloud_fractions.max():.2%})")
        logger.info(f"{'='*60}")
        
        # Recommendations
        mean_cloud = cloud_fractions.mean()
        if mean_cloud < 0.05:
            logger.warning(f"⚠️  SEVERE CLASS IMBALANCE: Only {mean_cloud:.2%} cloud pixels!")
            logger.warning(f"   Recommended pos_weight: {1.0/mean_cloud:.1f}")
        elif mean_cloud < 0.10:
            logger.warning(f"⚠️  MODERATE CLASS IMBALANCE: {mean_cloud:.2%} cloud pixels")
            logger.warning(f"   Recommended pos_weight: {1.0/mean_cloud:.1f}")
        elif mean_cloud < 0.20:
            logger.info(f"✓  Mild class imbalance: {mean_cloud:.2%} cloud pixels")
            logger.info(f"   Recommended pos_weight: {1.0/mean_cloud:.1f}")
        else:
            logger.info(f"✓  Good class balance: {mean_cloud:.2%} cloud pixels")
    
    return cloud_fractions


def compute_class_distribution(dataloader, max_batches=50):
    """
    Compute class distribution across entire dataset.
    
    Args:
        dataloader: DataLoader instance
        max_batches: Maximum number of batches to process
    """
    logger.info("Computing class distribution...")
    
    cloud_pixels = 0
    total_pixels = 0
    batch_count = 0
    
    for batch in dataloader:
        if batch is None:
            continue
        
        _, y = batch
        cloud_pixels += y.sum().item()
        total_pixels += y.numel()
        batch_count += 1
        
        if batch_count >= max_batches:
            break
    
    if total_pixels > 0:
        cloud_fraction = cloud_pixels / total_pixels
        logger.info(f"\n{'='*60}")
        logger.info(f"Class Distribution:")
        logger.info(f"  Cloud pixels: {cloud_pixels:,.0f}")
        logger.info(f"  Total pixels: {total_pixels:,.0f}")
        logger.info(f"  Cloud fraction: {cloud_fraction:.4f} ({cloud_fraction:.2%})")
        logger.info(f"  Recommended pos_weight: {1.0/cloud_fraction:.2f}")
        logger.info(f"{'='*60}")
        
        return cloud_fraction
    else:
        logger.warning("No valid pixels found!")
        return None


def check_label_quality(dataset, num_samples=20):
    """
    Check if cloud labels make sense visually.
    
    Args:
        dataset: MODISZarrPatchDataset instance
        num_samples: Number of samples to check
    """
    logger.info(f"Checking label quality on {num_samples} samples...")
    
    issues = []
    
    for i in range(min(num_samples, len(dataset))):
        try:
            x, y = dataset[i]
            
            red = x[0].numpy()
            nir = x[1].numpy()
            ndvi = x[2].numpy()
            cloud_mask = y.numpy()
            
            # Check for obvious issues
            
            # Issue 1: All pixels are cloud
            if cloud_mask.mean() > 0.95:
                issues.append(f"Sample {i}: Almost all pixels are cloud ({cloud_mask.mean():.2%})")
            
            # Issue 2: No cloud pixels at all
            if cloud_mask.mean() < 0.01:
                issues.append(f"Sample {i}: Almost no cloud pixels ({cloud_mask.mean():.2%})")
            
            # Issue 3: Cloud mask doesn't match NDVI
            # Clouds typically have high NDVI (vegetation) or low NDVI (snow/ice)
            # But should not have moderate NDVI (0.2-0.5)
            cloud_ndvi = ndvi[cloud_mask > 0.5]
            clear_ndvi = ndvi[cloud_mask < 0.5]
            
            if len(cloud_ndvi) > 0 and len(clear_ndvi) > 0:
                cloud_ndvi_mean = cloud_ndvi.mean()
                clear_ndvi_mean = clear_ndvi.mean()
                
                # If cloud NDVI is similar to clear NDVI, labels might be wrong
                if abs(cloud_ndvi_mean - clear_ndvi_mean) < 0.1:
                    issues.append(f"Sample {i}: Cloud and clear pixels have similar NDVI "
                                f"(cloud: {cloud_ndvi_mean:.3f}, clear: {clear_ndvi_mean:.3f})")
            
        except Exception as e:
            logger.warning(f"Error checking sample {i}: {e}")
            continue
    
    if issues:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"Found {len(issues)} potential label quality issues:")
        for issue in issues:
            logger.warning(f"  ⚠️  {issue}")
        logger.warning(f"{'='*60}")
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"✓  No obvious label quality issues found")
        logger.info(f"{'='*60}")
    
    return issues


def suggest_pos_weight(cloud_fraction):
    """
    Suggest optimal pos_weight based on class distribution.
    
    Args:
        cloud_fraction: Fraction of cloud pixels (0-1)
    """
    if cloud_fraction is None or cloud_fraction <= 0:
        logger.warning("Cannot compute pos_weight: invalid cloud fraction")
        return None
    
    # Standard pos_weight for class imbalance
    standard_weight = 1.0 / cloud_fraction
    
    # Adjusted weight (less aggressive)
    adjusted_weight = min(standard_weight, 20.0)  # Cap at 20
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Pos Weight Recommendations:")
    logger.info(f"  Cloud fraction: {cloud_fraction:.4f} ({cloud_fraction:.2%})")
    logger.info(f"  Standard pos_weight: {standard_weight:.2f}")
    logger.info(f"  Adjusted pos_weight: {adjusted_weight:.2f} (capped at 20)")
    logger.info(f"{'='*60}")
    
    return adjusted_weight


def run_full_diagnosis(dataset, dataloader, output_dir="diagnostics"):
    """
    Run complete diagnosis pipeline.
    
    Args:
        dataset: MODISZarrPatchDataset instance
        dataloader: DataLoader instance
        output_dir: Directory to save diagnostic outputs
    """
    logger.info("="*60)
    logger.info("RUNNING FULL DIAGNOSIS")
    logger.info("="*60)
    
    # Step 1: Visualize samples
    logger.info("\n[Step 1/4] Visualizing samples...")
    cloud_fractions = diagnose_cloud_labels(dataset, output_dir, num_samples=10)
    
    # Step 2: Check label quality
    logger.info("\n[Step 2/4] Checking label quality...")
    issues = check_label_quality(dataset, num_samples=20)
    
    # Step 3: Compute class distribution
    logger.info("\n[Step 3/4] Computing class distribution...")
    cloud_fraction = compute_class_distribution(dataloader, max_batches=50)
    
    # Step 4: Suggest pos_weight
    logger.info("\n[Step 4/4] Suggesting pos_weight...")
    suggested_weight = suggest_pos_weight(cloud_fraction)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("="*60)
    
    if issues:
        logger.warning(f"⚠️  Found {len(issues)} label quality issues")
        logger.warning("   → Consider fixing cloud labels before training")
    
    if cloud_fraction is not None:
        if cloud_fraction < 0.05:
            logger.warning(f"⚠️  Severe class imbalance ({cloud_fraction:.2%} cloud)")
            logger.warning(f"   → Increase pos_weight to {suggested_weight:.1f}")
        elif cloud_fraction < 0.10:
            logger.warning(f"⚠️  Moderate class imbalance ({cloud_fraction:.2%} cloud)")
            logger.warning(f"   → Increase pos_weight to {suggested_weight:.1f}")
        else:
            logger.info(f"✓  Reasonable class balance ({cloud_fraction:.2%} cloud)")
    
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDED ACTIONS:")
    logger.info("="*60)
    
    if issues:
        logger.info("1. Fix cloud labels (check MODIS state_1km bits)")
    
    if cloud_fraction is not None and cloud_fraction < 0.10:
        logger.info(f"2. Set pos_weight = {suggested_weight:.1f} in cloud_adapter.py")
    
    logger.info("3. Visualize samples in 'diagnostics/' folder")
    logger.info("4. Check if cloud masks match actual cloud patterns")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, 'src')
    
    from transform.cloud_adapter import CloudAdapterPipeline
    from utils import init_logging
    
    # Initialize logging
    logger = init_logging("diagnostics.log")
    
    # Create pipeline
    pipeline = CloudAdapterPipeline(
        model_name=("facebookresearch/dinov2", "dinov2_vits14")
    )
    
    # Load data (adjust paths as needed)
    from definitions import DATA_PATH
    path_modis_250 = DATA_PATH / "modis/data/modis" / "MOD09GQ_dataset.zarr"
    path_modis_500 = DATA_PATH / "modis/data/modis" / "MOD09GA_dataset.zarr"
    
    # Extract cubes
    from transform.cloud_adapter import extract_modis_cube
    cube_250 = extract_modis_cube(path_modis_250, "mod09_250", samples=50)
    cube_500 = extract_modis_cube(path_modis_500, "mod09_500", samples=50)
    
    # Create dataset
    from transform.cloud_adapter import MODISZarrPatchDataset
    dataset = MODISZarrPatchDataset(
        cube_250, cube_500,
        min_clear_fraction=0.05,
        dates=cube_250["dates"][:10]  # Use first 10 dates
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    from transform.cloud_adapter import safe_collate
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=safe_collate
    )
    
    # Run diagnosis
    run_full_diagnosis(dataset, dataloader, output_dir="diagnostics")
