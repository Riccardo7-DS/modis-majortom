from modis_majortom.utils.general import list_mtg_fci_objects, download_from_mtg_fci

# Non-recursive: just top-level prefixes, fast even for large Zarr stores
top_level = list_mtg_fci_objects(recursive=False)
matches = [obj for obj in top_level if "MTG_FCI_mtg_2025_06.zarr" in obj]

for prefix in matches:
    # Now recursively list only within this specific Zarr store prefix
    chunks = list_mtg_fci_objects(prefix=prefix, recursive=True)
    print(f"Found {len(chunks)} objects under {prefix}")
    for obj in chunks:
        download_from_mtg_fci(obj)


