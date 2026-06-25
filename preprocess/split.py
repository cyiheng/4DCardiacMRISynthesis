import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# File paths
DSB_TRAIN_CSV = ROOT_DIR / "data" / "DSB_nifti" / "train_metadata_additional_with_es.csv"
DSB_TEST_CSV = ROOT_DIR / "data" / "DSB_nifti" / "test_metadata_additional_with_es.csv"
ACDC_TRAIN_CSV = ROOT_DIR / "data" / "ACDC_Preprocessed" / "train_metadata.csv"
ACDC_TEST_CSV = ROOT_DIR / "data" / "ACDC_Preprocessed" / "test_metadata.csv"
DSB_BAD_CASES_CSV = ROOT_DIR / "data" / "bad_cases.csv"

OUTPUT_TRAIN_CSV = ROOT_DIR / "data" / "train_split_final.csv"
OUTPUT_TEST_CSV = ROOT_DIR / "data" / "test_split_final.csv"

def load_metadata(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["dataset"] = dataset_name
    df["metadata_file"] = str(csv_path)
    return df


def main():
    # 1. Load original metadata for ACDC and DSB
    print("Loading original metadata for ACDC and DSB...")
    df_acdc_train = load_metadata(ACDC_TRAIN_CSV, "ACDC")
    df_acdc_test = load_metadata(ACDC_TEST_CSV, "ACDC")
    df_dsb_train = load_metadata(DSB_TRAIN_CSV, "DSB")
    df_dsb_test = load_metadata(DSB_TEST_CSV, "DSB")
    
    # 2. Filter out bad cases from DSB splits only
    print("Removing bad DSB cases...")
    df_bad = pd.read_csv(DSB_BAD_CASES_CSV)
    bad_pids = set(df_bad['filter'].astype(str))
    
    df_dsb_train = df_dsb_train[~df_dsb_train['pid'].astype(str).isin(bad_pids)].copy()
    df_dsb_test = df_dsb_test[~df_dsb_test['pid'].astype(str).isin(bad_pids)].copy()

    # 3. Combine ACDC and DSB into final train/test splits
    df_train_final = pd.concat([df_acdc_train, df_dsb_train], ignore_index=True)
    df_test_final = pd.concat([df_acdc_test, df_dsb_test], ignore_index=True)

    # 4. Sanity checks
    print("-" * 30)
    print(f"ACDC train rows: {len(df_acdc_train)}")
    print(f"DSB train rows (after filtering): {len(df_dsb_train)}")
    print(f"Combined train rows: {len(df_train_final)}")
    print(f"ACDC test rows: {len(df_acdc_test)}")
    print(f"DSB test rows (after filtering): {len(df_dsb_test)}")
    print(f"Combined test rows: {len(df_test_final)}")
    print(f"Bad DSB cases excluded: {len(bad_pids)}")
    print("-" * 30)

    # 5. Save combined split files
    df_train_final.to_csv(OUTPUT_TRAIN_CSV, index=False)
    df_test_final.to_csv(OUTPUT_TEST_CSV, index=False)

    print(f"Saved train split: {OUTPUT_TRAIN_CSV}")
    print(f"Saved test split:  {OUTPUT_TEST_CSV}")

if __name__ == "__main__":
    main()