# check_brats_data.py
import os
import glob


def check_brats_structure():
    """Check the actual BraTS data structure"""
    base_path = "brats20-dataset-training-validation"

    # Check training data
    train_path = os.path.join(base_path, "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData")
    val_path = os.path.join(base_path, "BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData")

    print("🔍 Checking BraTS Data Structure...")

    for split_name, split_path in [("Training", train_path), ("Validation", val_path)]:
        print(f"\n{split_name} Data:")

        if not os.path.exists(split_path):
            print(f"  ❌ Path doesn't exist: {split_path}")
            continue

        # List all case directories
        case_dirs = [d for d in os.listdir(split_path)
                     if os.path.isdir(os.path.join(split_path, d))]

        print(f"  Found {len(case_dirs)} cases")

        if case_dirs:
            # Check first case
            first_case = case_dirs[0]
            case_path = os.path.join(split_path, first_case)
            files = os.listdir(case_path)

            print(f"  First case: {first_case}")
            print(f"  Files: {files}")

            # Check for required files
            has_t1 = any('t1' in f.lower() for f in files)
            has_t1ce = any('t1ce' in f.lower() for f in files)
            has_t2 = any('t2' in f.lower() for f in files)
            has_flair = any('flair' in f.lower() for f in files)
            has_seg = any('seg' in f.lower() for f in files)

            print(f"  Modalities - T1: {has_t1}, T1ce: {has_t1ce}, T2: {has_t2}, FLAIR: {has_flair}, SEG: {has_seg}")


if __name__ == '__main__':
    check_brats_structure()