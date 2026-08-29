"""
Notebook/Step 00: Project Setup and Reproducibility.

CRISP-DM: pre-Business/Data-Understanding project setup.

This step:
1. Installs/checks the required Python package versions.
2. Sets reproducibility seeds.
3. Creates the required project directories.
4. Verifies that the raw AMLNet dataset is present.
5. Optionally verifies the dataset MD5 checksum.
6. Records the software environment for reproducibility.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


# ============================================================
# 1. Required package versions
# ============================================================

REQUIRED_PACKAGES = {
    "numpy": "2.4.6",
    "pandas": "3.0.5",
    "scikit-learn": "1.9.0",
    "matplotlib": "3.11.1",
    "seaborn": None,
    "shap": "0.52.0",
    "xgboost": "3.3.0",
    "lightgbm": None,
    "imbalanced-learn": "0.14.2",
    "joblib": "1.5.3",
    "pyarrow": "24.0.0",
    "streamlit": "1.60.0",
}


# ============================================================
# 2. Install required packages
# ============================================================

def install_required_packages() -> None:
    """Install the required package versions using this Python interpreter."""

    print("\n[AMLNet] Checking required Python packages...")

    packages = []

    for package, version in REQUIRED_PACKAGES.items():
        if version is None:
            packages.append(package)
        else:
            packages.append(f"{package}=={version}")

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
    ]

    command.extend(packages)

    print("\n[AMLNet] Installing/checking packages:")
    for package in packages:
        print(f"  - {package}")

    result = subprocess.run(
        command,
        check=False,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "\n[AMLNet] Package installation failed.\n"
            "Please check the pip error messages above."
        )

    print("\n[AMLNet] Package installation/check completed successfully.")


# ============================================================
# 3. Import project dependencies AFTER installation
# ============================================================

def load_dependencies():
    """Import dependencies after required packages have been installed."""

    global pd
    global C

    import pandas as pd

    sys.path.append(
        str(Path(__file__).resolve().parent)
    )

    import amlnet_common as C


# ============================================================
# 4. MD5 checksum
# ============================================================

def md5_of_file(path: Path, chunk_mb: int = 8) -> str:
    """Calculate the MD5 checksum of a file."""

    h = hashlib.md5()

    with open(path, "rb") as fh:
        for chunk in iter(
            lambda: fh.read(chunk_mb * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# 5. Main
# ============================================================

def main(verify_md5: bool = False) -> None:

    # --------------------------------------------------------
    # Install/check packages first
    # --------------------------------------------------------

    install_required_packages()

    # Import pandas and project module only after installation
    load_dependencies()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    C.set_seeds()

    # --------------------------------------------------------
    # Create required directories
    # --------------------------------------------------------

    C.ensure_dirs()

    C.log_step(
        f"Project root: {C.PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # Folder structure evidence
    # --------------------------------------------------------

    folder_summary = pd.DataFrame(
        {
            "folder": [
                d.name
                for d in C.REQUIRED_DIRS
            ],
            "path": [
                str(d)
                for d in C.REQUIRED_DIRS
            ],
            "exists": [
                d.exists()
                for d in C.REQUIRED_DIRS
            ],
        }
    )

    C.save_table(
        folder_summary,
        "project_folder_structure",
    )

    # --------------------------------------------------------
    # Dataset availability check
    # --------------------------------------------------------

    exists = C.RAW_DATA_PATH.exists()

    check = {
        "expected_dataset_filename": C.DATASET_FILENAME,
        "expected_dataset_location": str(
            C.RAW_DATA_PATH
        ),
        "dataset_doi": C.DATASET_DOI,
        "dataset_file_exists": exists,
        "file_size_mb": (
            round(
                C.RAW_DATA_PATH.stat().st_size
                / 1024 ** 2,
                2,
            )
            if exists
            else None
        ),
        "expected_md5": C.DATASET_MD5,
        "md5_verified": None,
    }

    # --------------------------------------------------------
    # Optional MD5 verification
    # --------------------------------------------------------

    if exists and verify_md5:

        actual = md5_of_file(
            C.RAW_DATA_PATH
        )

        check["md5_verified"] = (
            actual == C.DATASET_MD5
        )

        check["actual_md5"] = actual

    # --------------------------------------------------------
    # Dataset status
    # --------------------------------------------------------

    if not exists:

        C.log_step(
            "AMLNet dataset NOT found. "
            "Download from "
            "https://zenodo.org/records/16736515 "
            "and save as "
            f"{C.RAW_DATA_PATH}"
        )

    else:

        C.log_step(
            f"AMLNet dataset found "
            f"({check['file_size_mb']} MB)."
        )

    C.save_table(
        pd.DataFrame([check]),
        "dataset_availability_check",
    )

    # --------------------------------------------------------
    # Environment information
    # --------------------------------------------------------

    env = C.environment_summary()

    C.save_table(
        env,
        "environment_and_package_versions",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    C.log_step(
        "Step 00 complete. "
        "Evidence tables written to outputs/tables/."
    )

    print("\n" + env.to_string(index=False))


# ============================================================
# 6. Run Step 00
# ============================================================

if __name__ == "__main__":

    main(
        verify_md5="--verify-md5" in sys.argv
    )