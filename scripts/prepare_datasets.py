from pathlib import Path
import argparse
import csv
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import kagglehub

import requests
from PIL import Image
from tqdm import tqdm


# CAC DUONG DAN
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOWNLOAD_DIR = ROOT / "downloads"

COCO_DIR = DATA_DIR / "coco"
IMGNET_DIR = DATA_DIR / "imgnet"
CIRR_DIR = DATA_DIR / "CIRR"
FASHION_DIR = DATA_DIR / "fashion-iq"

CC3M_RAW_DIR = DOWNLOAD_DIR / "cc3m" / "raw"
CC3M_DATASET_ID = "ad271828/conceptual-captions-dataset-train-and-validation"
CC3M_TRAIN_NAME = "Train_GCC-training.tsv"
CC3M_VAL_NAME = "Validation_GCC-1.1.0-Validation.tsv"

CIRR_KAGGLE_DATASET_ID = "anhngzv/cirr-cir"

CIRR_KAGGLE_SELECTED_DIR = DOWNLOAD_DIR / "cirr_kaggle_selected"
CIRR_KAGGLE_FULL_DIR = DOWNLOAD_DIR / "cirr_kaggle_full"

# TAO CAU TRUC THU MUC
def create_directories():
    directories = [
        ROOT / "cc",
        ROOT / "cc_data" / "train",
        ROOT / "cc_data" / "val",
        CC3M_RAW_DIR,
        DOWNLOAD_DIR / "coco",
        DOWNLOAD_DIR / "imagenet",
        COCO_DIR / "annotations",
        COCO_DIR / "val2017",
        COCO_DIR / "val2017_masked",
        IMGNET_DIR / "real",
        IMGNET_DIR / "imagenet-r",
        CIRR_DIR / "captions",
        CIRR_DIR / "image_splits",
        CIRR_DIR / "dev",
        CIRR_DIR / "test1",
        FASHION_DIR / "json",
        FASHION_DIR / "image_splits",
        FASHION_DIR / "images",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(" Da tao/kiem tra cac thu muc.", flush=True)

# TAI MOT FILE TU URL
def download_file(url, destination, retries=5):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = destination.with_name(destination.name + ".part")

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Da co file: {destination.name}", flush=True)
        return destination

    print(f"\n{destination.name}", flush=True)
    print(f"{url}", flush=True)
    print(f"{destination}", flush=True)

    for attempt in range(1, retries + 1):
        try:
            downloaded_size = (
                temporary_file.stat().st_size
                if temporary_file.exists()
                else 0
            )

            headers = {}
            if downloaded_size > 0:
                headers["Range"] = f"bytes={downloaded_size}-"

            print(
                f"Lan thu {attempt}/{retries} | "
                f"Da co: {downloaded_size / (1024 ** 3):.2f} GB",
                flush=True,
            )

            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(20, 120),
            ) as response:

                print(f"Status: {response.status_code}", flush=True)

                # Neu server ho tro resume, phai tra ve 206.
                if downloaded_size > 0 and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    expected_prefix = f"bytes {downloaded_size}-"

                    if not content_range.startswith(expected_prefix):
                        raise RuntimeError(
                            f"Content-Range khong hop le: {content_range}"
                        )

                    mode = "ab"
                    total_size = int(content_range.rsplit("/", 1)[1]) \
                        if "/" in content_range and \
                        content_range.rsplit("/", 1)[1].isdigit() else None

                else:
                    # Server khong ho tro Range hoac tra ve 200:
                    # phai tai lai tu dau, khong noi du lieu sai.
                    mode = "wb"
                    downloaded_size = 0

                    content_length = response.headers.get("Content-Length")
                    total_size = (
                        int(content_length)
                        if content_length and content_length.isdigit()
                        else None
                    )

                response.raise_for_status()

                if total_size is not None:
                    print(
                        f"Tong kich thuoc: {total_size / (1024 ** 3):.2f} GB",
                        flush=True,
                    )

                with open(temporary_file, mode) as file:
                    with tqdm(
                        total=total_size,
                        initial=downloaded_size,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=destination.name,
                        dynamic_ncols=True,
                    ) as progress:

                        for chunk in response.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                file.write(chunk)
                                progress.update(len(chunk))

                actual_size = temporary_file.stat().st_size

                if total_size is not None and actual_size != total_size:
                    raise RuntimeError(
                        f"File chua day du: {actual_size} bytes, "
                        f"can {total_size} bytes."
                    )

                if actual_size == 0:
                    raise RuntimeError("File tai ve bi rong.")

                temporary_file.replace(destination)

                print(
                    f"\nDa tai xong: {destination.name}",
                    flush=True,
                )
                return destination

        except Exception as error:
            print(f"\nLan {attempt} that bai: {error}", flush=True)

            # QUAN TRONG: khong xoa file .part.
            # Lan tiep theo se tiep tuc tu kich thuoc hien tai.
            if attempt < retries:
                import time
                time.sleep(5)

    raise RuntimeError(
        f"Khong tai duoc {destination.name} sau {retries} lan. "
        f"Du lieu tam duoc giu tai: {temporary_file}"
    )

# GIAI NEN ZIP
def extract_zip(archive_path, destination):
    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            extracted_path = (destination / member.filename).resolve()
            if not extracted_path.is_relative_to(destination_root):
                raise ValueError(f"Duong dan khong an toan trong ZIP: {member.filename}")
        archive.extractall(destination)

    print(f" Da giai nen: {archive_path}", flush=True)

def safe_extract_tar(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    root = destination.resolve()

    with tarfile.open(archive, "r:*") as tar_file:
        for item in tar_file.getmembers():
            target = (destination / item.name).resolve()

            if not target.is_relative_to(root):
                raise ValueError(f"Đường dẫn không an toàn: {item.name}")

        tar_file.extractall(destination)

# CC3M: TAI DANH SACH TSV TU KAGGLEHUB
def download_cc3m_kaggle():
    if kagglehub is None:
        raise ImportError("Chua cai kagglehub. Hay chay: python -m pip install kagglehub")

    expected_files = [
        CC3M_RAW_DIR / CC3M_TRAIN_NAME,
        CC3M_RAW_DIR / CC3M_VAL_NAME,
    ]

    # Neu hai TSV da co, khong tai lai dataset.
    if all(path.is_file() and path.stat().st_size > 0 for path in expected_files):
        print("Hai file TSV CC3M da ton tai.", flush=True)
        dataset_path = CC3M_RAW_DIR
    else:
        print("\nTai CC3M tu KaggleHub...", flush=True)
        dataset_path = Path(
            kagglehub.dataset_download(
                CC3M_DATASET_ID,
                output_dir=str(CC3M_RAW_DIR),
            )
        )

    print(f"Duong dan dataset: {dataset_path}", flush=True)
    files = sorted(path for path in dataset_path.rglob("*") if path.is_file())

    if not files:
        raise FileNotFoundError(f"Khong tim thay file nao trong {dataset_path}")

    print("\nCac file da tai", flush=True)
    for file in files:
        size_mb = file.stat().st_size / (1024 ** 2)
        print(f"{file.relative_to(dataset_path)} | {size_mb:.2f} MB", flush=True)

    # Tim theo ten file neu KaggleHub dat file trong thu muc con.
    for expected in expected_files:
        if not expected.exists():
            matches = list(dataset_path.rglob(expected.name))
            if matches:
                expected.parent.mkdir(parents=True, exist_ok=True)
                if matches[0].resolve() != expected.resolve():
                    shutil.copy2(matches[0], expected)

    missing = [str(path) for path in expected_files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        print("\nKhong tim thay du ca hai file TSV mong doi:", flush=True)
        for path in missing:
            print(f"  - {path}", flush=True)
        print("Hay kiem tra cac ten file trong danh sach ben tren.", flush=True)
    else:
        print("\n Da co TSV train va validation CC3M.", flush=True)

    return dataset_path

# CC3M: doc file TSV
def read_cc3m_tsv(tsv_path, limit=0):
    tsv_path = Path(tsv_path)
    if not tsv_path.is_file():
        raise FileNotFoundError(f"Khong tim thay file TSV: {tsv_path}")

    valid_count = 0
    with open(tsv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row_number, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue

            first = row[0].strip()
            second = row[1].strip()

            if first.lower() in {"caption", "text"} and second.lower() in {"url", "image_url", "image url"}:
                continue

            if first.startswith(("http://", "https://")):
                image_url, caption = first, second
            elif second.startswith(("http://", "https://")):
                caption, image_url = first, second
            else:
                continue

            if not caption or not image_url:
                continue

            yield row_number, caption, image_url
            valid_count += 1
            if limit > 0 and valid_count >= limit:
                return

# CC3M
def download_cc3m_image(record, split):
    row_number, caption, image_url = record
    image_dir = ROOT / "cc_data" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{split}_{row_number:09d}"
    temporary_path = image_dir / f"{stem}.part"
    extensions = {
        "JPEG": ".jpg",
        "JPG": ".jpg",
        "PNG": ".png",
        "WEBP": ".webp",
        "GIF": ".gif",
        "BMP": ".bmp",
        "TIFF": ".tiff",
    }

    for existing in image_dir.glob(stem + ".*"):
        if existing.suffix == ".part":
            continue
        try:
            with Image.open(existing) as image:
                image.verify()
            return record, existing, None
        except Exception:
            existing.unlink(missing_ok=True)

    last_error = None
    headers = {"User-Agent": "Pic2Word-CC3M-Dataset-Preparation/1.0"}

    for attempt in range(5):
        try:
            with requests.get(image_url,headers=headers, stream=True,timeout=(15, 60)) as response:
                response.raise_for_status()
                with open(temporary_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            file.write(chunk)

            with Image.open(temporary_path) as image:
                image_format = image.format
                image.verify()

            extension = extensions.get(image_format)
            if extension is None:
                raise ValueError(f"Dinh dang anh khong ho tro: {image_format}")

            destination = image_dir / f"{stem}{extension}"
            temporary_path.replace(destination)
            return record, destination, None

        except Exception as error:
            temporary_path.unlink(missing_ok=True)
            last_error = str(error)
            if attempt < 2:
                time.sleep(attempt + 1)

    return record, None, last_error or "Tai anh that bai."

# CC3M: Tao CSV input PIC2WORD
def prepare_cc3m_split(source_file, split, limit=0, workers=4):
    cc_dir = ROOT / "cc"
    log_dir = DOWNLOAD_DIR / "cc3m"
    cc_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if split == "train":
        output_csv = cc_dir / "Train_GCC-training_output.csv"
    else:
        output_csv = cc_dir / "Validation_GCC-1.1.0-Validation_output.csv"

    temporary_csv = Path(str(output_csv) + ".part")
    failed_file = log_dir / f"{split}_failed.tsv"

    successful = 0
    failed = 0
    processed = 0
    rows = read_cc3m_tsv(source_file, limit=limit)

    def batches(iterator, batch_size):
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    with (open(temporary_csv, "w", encoding="utf-8", newline="") as csv_file,open(failed_file, "w", encoding="utf-8", newline="") as error_file):
        writer = csv.DictWriter(csv_file, fieldnames=["filepath", "title"], delimiter="\t")
        writer.writeheader()
        error_writer = csv.writer(error_file, delimiter="\t")
        error_writer.writerow(["row_number", "url", "error"])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            with tqdm(
                total=limit if limit > 0 else None,
                desc=f"CC3M {split}",
                unit="img",
                dynamic_ncols=True,
            ) as progress:
                for batch in batches(rows, workers * 4):
                    results = executor.map(
                        lambda record: download_cc3m_image(record, split),
                        batch,
                    )

                    # executor.map giu thu tu ban ghi, CSV se on dinh.
                    for record, image_path, error in results:
                        row_number, caption, image_url = record
                        processed += 1

                        if image_path is None:
                            error_writer.writerow([row_number, image_url, error])
                            failed += 1
                        else:
                            relative_path = image_path.relative_to(ROOT).as_posix()
                            writer.writerow({
                                "filepath": relative_path,
                                "title": caption,
                            })
                            successful += 1

                        progress.update(1)
                        progress.set_postfix(success=successful, failed=failed)

                    csv_file.flush()
                    error_file.flush()

    temporary_csv.replace(output_csv)
    print(f"\n Hoan tat CC3M {split}.", flush=True)
    print(f"  Da xu ly: {processed}", flush=True)
    print(f"  Anh hop le: {successful}", flush=True)
    print(f"  Anh loi: {failed}", flush=True)
    print(f"  CSV: {output_csv}", flush=True)
    print(f"  Log loi: {failed_file}", flush=True)


def prepare_cc3m(limit=0, workers=4):
    if limit < 0:
        raise ValueError("--limit phai >= 0")
    if workers < 1:
        raise ValueError("--workers phai >= 1")

    train_file = CC3M_RAW_DIR / CC3M_TRAIN_NAME
    val_file = CC3M_RAW_DIR / CC3M_VAL_NAME

    if not train_file.is_file() or not val_file.is_file():
        raise FileNotFoundError(
            "Chua tim thay TSV CC3M. Chay --cc3m-kaggle truoc va kiem tra ten file trong downloads/cc3m/raw."
        )
    prepare_cc3m_split(train_file, split="train", limit=limit, workers=workers)
    prepare_cc3m_split(val_file, split="val", limit=limit, workers=workers)

# COCO
def prepare_coco():
    coco_downloads = DOWNLOAD_DIR / "coco"
    image_zip = coco_downloads / "val2017.zip"
    annotation_zip = coco_downloads / "annotations_trainval2017.zip"
    image_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if not any(COCO_DIR.joinpath("val2017").glob("*.jpg")):
        print("Tai COCO validation images.", flush=True)
        download_file(image_url, image_zip)
        print("Giai nen COCO validation images.", flush=True)
        extract_zip(image_zip, COCO_DIR)
    else:
        print("COCO validation images da ton tai.", flush=True)

    annotation_json = COCO_DIR / "annotations" / "instances_val2017.json"
    if not annotation_json.exists():
        print("Tai COCO annotations.", flush=True)
        download_file(annotation_url, annotation_zip)
        extract_zip(annotation_zip, COCO_DIR)
    else:
        print("COCO annotation da ton tai.", flush=True)

    prepare_script = COCO_DIR / "prepare_data.py"

    if not prepare_script.exists():
        download_file(
            "https://raw.githubusercontent.com/google-research/composed_image_retrieval/main/data/coco/prepare_data.py",
            prepare_script
        )

    # prepare_script = COCO_DIR / "prepare_data.py"
    # if not prepare_script.exists():
    #     raise FileNotFoundError(
    #         "Khong tim thay data/coco/prepare_data.py trong repository."
    #     )

    print("Tao COCO mask va coco_eval.csv.", flush=True)
    subprocess.run([sys.executable, str(prepare_script)], cwd=COCO_DIR, check=True)
    print(" Da chuan bi COCO.", flush=True)

# IMAGENET-R
def prepare_imagenet_r():
    archive_path = DOWNLOAD_DIR / "imagenet" / "imagenet-r.tar"
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    target_dir = IMGNET_DIR / "imagenet-r"

    if target_dir.exists() and any(target_dir.iterdir()):
        print("ImageNet-R da duoc giai nen.", flush=True)
        return

    download_file(url, archive_path)
    IMGNET_DIR.mkdir(parents=True, exist_ok=True)
    root = IMGNET_DIR.resolve()

    for filename in ["imgnet_real_query.txt","imgnet_targets.txt"]:
        destination = IMGNET_DIR / filename

        if not destination.exists():
            download_file(f"https://raw.githubusercontent.com/google-research/composed_image_retrieval/main/data/imgnet/{filename}",destination)
            
    safe_extract_tar(archive_path,IMGNET_DIR)

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in list(IMGNET_DIR.iterdir()):
            if item.is_dir() and item.name.startswith("n") and item.name[1:].isdigit():
                shutil.move(str(item), str(target_dir / item.name))

    if not target_dir.exists() or not any(target_dir.iterdir()):
        raise RuntimeError("Khong tim thay cac thu muc class cua ImageNet-R sau khi giai nen.")

    print(" Da chuan bi ImageNet-R.", flush=True)

# FASHION-IQ metadata
def prepare_fashioniq_metadata():
    repo_zip = (DOWNLOAD_DIR/ "fashion-iq"/ "fashion-iq-official.zip")
    extract_dir = (DOWNLOAD_DIR/ "fashion-iq"/ "official_source")

    caption_dir = FASHION_DIR / "json"
    split_dir = FASHION_DIR / "image_splits"

    caption_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    # Tải repository chính thức của Fashion-IQ
    download_file("https://github.com/XiaoxiaoGuo/fashion-iq/archive/refs/heads/master.zip",repo_zip)

    # Giải nén nếu chưa tìm được các thư mục nguồn
    source_roots = []

    if extract_dir.exists():
        source_roots = [
            path.parent
            for path in extract_dir.rglob("captions")
            if path.is_dir() and (path.parent / "image_splits").is_dir()
        ]

    if not source_roots:
        extract_zip(repo_zip, extract_dir)

        source_roots = [
            path.parent
            for path in extract_dir.rglob("captions")
            if path.is_dir() and (path.parent / "image_splits").is_dir()
        ]

    if not source_roots:
        raise FileNotFoundError("Không tìm thấy cặp thư mục captions/và image_splits/ trong repository Fashion-IQ.")

    source_root = source_roots[0]

    # Sao chép tất cả file JSON caption
    copied_captions = 0

    for source in (source_root / "captions").glob("*.json"):
        shutil.copy2(source,caption_dir / source.name)
        copied_captions += 1

    # Sao chép tất cả file JSON phân chia dữ liệu
    copied_splits = 0

    for source in (source_root / "image_splits").glob("*.json"):
        shutil.copy2(source,split_dir / source.name)
        copied_splits += 1

    if copied_captions == 0 or copied_splits == 0:
        raise RuntimeError("Không sao chép được đầy đủ caption và image split.")

    print("Đã chuẩn bị annotation Fashion-IQ.")
    print(f"  Caption files: {copied_captions}")
    print(f"  Split files:   {copied_splits}")
    print(f"  Captions:      {caption_dir}")
    print(f"  Image splits:  {split_dir}")

def fashion_image_is_valid(path):
    path = Path(path)

    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True

    except Exception:
        return False


def prepare_fashioniq_image_metadata():
    repo_zip = (DOWNLOAD_DIR/ "fashion-iq"/ "fashion-iq-metadata.zip")
    extract_dir = (DOWNLOAD_DIR/ "fashion-iq"/ "image_metadata")
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Nếu mapping đã được giải nén thì dùng lại
    mapping_files = list(extract_dir.rglob("asin2url.*.txt"))

    if not mapping_files:
        download_file("https://github.com/hongwang600/fashion-iq-metadata/archive/refs/heads/master.zip",repo_zip)
        extract_zip(repo_zip, extract_dir)
        mapping_files = list(extract_dir.rglob("asin2url.*.txt"))
            
    if not mapping_files:
        raise FileNotFoundError("Không tìm thấy asin2url.dress.txt, asin2url.shirt.txt hoặc asin2url.toptee.txt.")

    # Tất cả mapping thuộc cùng thư mục image_url
    mapping_dir = mapping_files[0].parent

    expected = {"asin2url.dress.txt","asin2url.shirt.txt","asin2url.toptee.txt",}
    found = {path.name for path in mapping_dir.glob("asin2url.*.txt")}

    missing = expected - found

    if missing:
        raise FileNotFoundError(
            "Thiếu file mapping: "
            + ", ".join(sorted(missing))
        )

    print(f"Fashion-IQ image metadata: {mapping_dir}")

    return mapping_dir


def download_one_fashioniq_image(record):
    asin, image_url, broken_dir = record

    image_dir = FASHION_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    destination = image_dir / f"{asin}.png"

    if fashion_image_is_valid(destination):
        return asin, True, None

    temp_file = image_dir / f"{asin}.download"
    temp_png = image_dir / f"{asin}.part.png"

    backup_image = broken_dir / f"{asin}.jpg"

    if fashion_image_is_valid(backup_image):
        try:
            with Image.open(backup_image) as image:
                image.load()
                image.convert("RGB").save(temp_png,format="PNG")

            temp_png.replace(destination)
            return asin, True, None

        except Exception:
            temp_png.unlink(missing_ok=True)

    last_error = None
    for attempt in range(3):
        try:
            with requests.get(image_url,stream=True,timeout=(15, 60),headers={"User-Agent": "Mozilla/5.0"}) as response:
                response.raise_for_status()
                with open(temp_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            file.write(chunk)

            with Image.open(temp_file) as image:
                image.load()
                image.convert("RGB").save(temp_png,format="PNG")

            temp_png.replace(destination)
            temp_file.unlink(missing_ok=True)

            return asin, True, None

        except Exception as error:
            last_error = str(error)

            temp_file.unlink(missing_ok=True)
            temp_png.unlink(missing_ok=True)

            if attempt < 2:
                time.sleep(attempt + 1)

    return asin, False, last_error or "Không tải được ảnh."

def prepare_fashioniq_images(limit=0, workers=4):
    create_directories()

    # Tải và giải nén metadata của ảnh
    mapping_dir = prepare_fashioniq_image_metadata()
    broken_dir = mapping_dir / "broken_links"

    categories = ["dress", "shirt", "toptee"]

    # Dùng dict để tránh tải lặp cùng một ASIN
    records = {}

    for category in categories:
        mapping_file = (mapping_dir / f"asin2url.{category}.txt")

        with open(mapping_file,"r",encoding="utf-8-sig",errors="replace") as file:
            for line in file:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2:
                    continue
                asin = parts[0].strip()
                image_url = parts[1].strip()
                if not asin or not image_url:
                    continue
                records.setdefault(asin,(asin, image_url, broken_dir))

    # Chỉ tải ảnh chưa tồn tại hoặc ảnh cũ bị hỏng
    todo = []

    for asin, record in records.items():
        image_path = (FASHION_DIR / "images" / f"{asin}.png")

        if not fashion_image_is_valid(image_path):
            todo.append(record)

    if limit > 0:
        todo = todo[:limit]

    print(f"Số ASIN trong mapping: {len(records)}")
    print(f"Số ảnh cần xử lý: {len(todo)}")

    failed_file = (DOWNLOAD_DIR/ "fashion-iq"/ "failed_images.tsv")
    failed_file.parent.mkdir(parents=True,exist_ok=True)

    success = 0
    failed = 0

    with open(failed_file,"w",encoding="utf-8",newline="") as error_file:
        error_writer = csv.writer(error_file,delimiter="\t")
        error_writer.writerow([ "asin","url","error"])

        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(download_one_fashioniq_image,todo)

            for asin, ok, error in tqdm(results,total=len(todo),desc="Fashion-IQ images",unit="img",dynamic_ncols=True):
                if ok:
                    success += 1
                else:
                    failed += 1
                    error_writer.writerow([asin,records[asin][1],error])
    print(f"ASIN đã đọc:       {len(records)}")
    print(f"Ảnh xử lý lượt này: {len(todo)}")
    print(f"Thành công:         {success}")
    print(f"Thất bại:           {failed}")
    print(f"Thư mục ảnh:        {FASHION_DIR / 'images'}")
    print(f"File lỗi:           {failed_file}")


# CIRR metadata
def prepare_cirr_metadata():
    source_zip = DOWNLOAD_DIR / "CIRR_annotations.zip"
    extract_dir = DOWNLOAD_DIR / "CIRR_annotations_source"

    CIRR_DIR.mkdir(parents=True, exist_ok=True)
    required_dirs = ["captions","captions_ext","image_splits",]

    def find_source_root():
        if not extract_dir.exists():
            return None
        for captions_dir in extract_dir.rglob("captions"):
            candidate = captions_dir.parent
            if not (candidate / "image_splits").is_dir():
                continue
            return candidate
        return None
    
    source_root = find_source_root()
    # Tải và giải nén nếu chưa có annotation nguồn
    if source_root is None:
        download_file("https://github.com/Cuberick-Orion/CIRR/archive/refs/heads/cirr_dataset.zip",source_zip)

        extract_zip(source_zip, extract_dir)
        source_root = find_source_root()

    if source_root is None:
        raise FileNotFoundError("Không tìm thấy thư mục captions và image_splits trong gói annotation CIRR.")

    # Sao chép annotation sang thư mục Pic2Word
    for folder_name in required_dirs:
        source = source_root / folder_name
        destination = CIRR_DIR / folder_name

        if not source.is_dir():
            if folder_name == "captions_ext":
                print( "Repository không có captions_ext; bỏ qua thư mục annotation bổ sung.")
                continue
            raise FileNotFoundError(f"Không tìm thấy thư mục annotation: {source}")

        shutil.copytree(source,destination,dirs_exist_ok=True)
        print(f"{folder_name}: {destination}")
    required_files = [CIRR_DIR / "captions" / "cap.rc2.val.json",CIRR_DIR / "image_splits" / "split.rc2.val.json"]

    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Thiếu annotation validation cần thiết: {path}")

    print("Đã chuẩn bị annotation CIRR.")

def prepare_cirr_raw_images(archive_path):
    archive_path = Path(archive_path)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy gói ảnh CIRR: {archive_path}")

    staging_dir = DOWNLOAD_DIR / "CIRR_raw_extracted"
    staging_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        extract_zip(archive_path, staging_dir)

    elif tarfile.is_tarfile(archive_path):
        safe_extract_tar(archive_path, staging_dir)

    else:
        raise ValueError("Gói ảnh phải là ZIP, TAR hoặc TAR.GZ.")
    image_root = None

    for dev_dir in staging_dir.rglob("dev"):
        if not dev_dir.is_dir():
            continue

        candidate = dev_dir.parent

        if (candidate / "test1").is_dir():
            image_root = candidate
            break

    if image_root is None:
        raise FileNotFoundError("Không tìm được thư mục có cả dev/ và test1/trong gói ảnh.Hãy kiểm tra gói đã tải và cấu trúc sau giải nén.")
    def merge_directory(source, destination):
        destination.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            target = destination / item.name

            if item.is_dir():
                if target.exists() and target.is_dir():
                    merge_directory(item, target)
                    if item.exists():
                        shutil.rmtree(item)
                else:
                    shutil.move(str(item), str(target))
            else:
                if target.exists():
                    item.unlink()
                else:
                    shutil.move(str(item), str(target))

    for split in ["train", "dev", "test1"]:
        source = image_root / split
        destination = CIRR_DIR / split

        if not source.is_dir():
            if split == "train":
                print("Gói ảnh không có train/. Ảnh train có thể không đầy đủ.")
                continue

            raise FileNotFoundError(f"Không tìm thấy thư mục {split}/ trong gói ảnh.")
        merge_directory(source, destination)
        print(f"Đã chuẩn bị ảnh {split}: {destination}")

    # Kiểm tra ảnh validation và test
    for split in ["dev", "test1"]:
        folder = CIRR_DIR / split

        image_count = sum(
            1 for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg" }
        ) if folder.exists() else 0
        print(f"{split}: {image_count} ảnh")
        if image_count == 0:
            raise RuntimeError(f"Thư mục {folder} không có ảnh.")
    print("Đã tải CIRR")

def find_cirr_image_root(search_roots):
    """
    Tim thu muc goc co dev/ va test1/.
    train/ la thu muc tuy chon.
    Ho tro CIRR nam long trong nhieu cap thu muc.
    """
    for root in search_roots:
        root = Path(root)

        if not root.is_dir():
            continue

        # Truong hop root chinh la thu muc chua cac split
        candidates = [root]

        # Tim cac thu muc dev/ ben trong root
        candidates.extend(
            path.parent
            for path in root.rglob("dev")
            if path.is_dir()
        )

        for candidate in candidates:
            if (
                (candidate / "dev").is_dir()
                and (candidate / "test1").is_dir()
            ):
                return candidate

    return None


def find_cirr_annotation_root(search_roots):
    """
    Tim thu muc co ca captions/ va image_splits/.
    Vi du:
        CIRR/captions/
        CIRR/image_splits/

    Hoac:
        CIRR/cirr/captions/
        CIRR/cirr/image_splits/
    """
    for root in search_roots:
        root = Path(root)

        if not root.is_dir():
            continue

        candidates = [root]
        candidates.extend(
            path.parent
            for path in root.rglob("captions")
            if path.is_dir()
        )

        for candidate in candidates:
            if (
                (candidate / "captions").is_dir()
                and (candidate / "image_splits").is_dir()
            ):
                return candidate

    return None


def copy_merge_directory(source, destination):
    """
    Sao chep de quy, giu nguyen cau truc thu muc.
    Khong di chuyen hoac xoa du lieu goc da tai tu Kaggle.
    Bo qua file dich da co va co cung kich thuoc.
    """
    source = Path(source)
    destination = Path(destination)

    if not source.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay thu muc nguon: {source}"
        )

    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = destination / item.name

        if item.is_dir():
            copy_merge_directory(item, target)

        elif item.is_file():
            if (
                not target.is_file()
                or item.stat().st_size != target.stat().st_size
            ):
                shutil.copy2(item, target)


def count_cirr_images(folder):
    """Dem anh trong mot split CIRR."""
    folder = Path(folder)

    if not folder.is_dir():
        return 0

    extensions = {".png", ".jpg", ".jpeg", ".webp"}

    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
    )


def prepare_cirr_kaggle():
    if kagglehub is None:
        raise ImportError("Chua cai kagglehub. Chay: python -m pip install --upgrade kagglehub")

    create_directories()

    search_roots = []

    print("\nTai CIRR tu Kaggle...", flush=True)
    print(f"Dataset: {CIRR_KAGGLE_DATASET_ID}",flush=True)

    try:
        selected_path = Path(
            kagglehub.dataset_download(
                CIRR_KAGGLE_DATASET_ID,
                path="CIRR",
                output_dir=str(CIRR_KAGGLE_SELECTED_DIR),
            )
        )

        search_roots.append(selected_path)

        print(f"Duong dan CIRR da tai: {selected_path}",flush=True)

    except Exception as error:
        print("Khong tai rieng duoc thu muc CIRR bang phien ban KaggleHub hien tai.",flush=True)
        print(f"Chi tiet: {error}", flush=True)

    image_root = find_cirr_image_root(search_roots)

    if image_root is None:
        print("\nThu tai toan bo dataset Kaggle...",flush=True)

        full_path = Path(kagglehub.dataset_download(CIRR_KAGGLE_DATASET_ID,output_dir=str(CIRR_KAGGLE_FULL_DIR),))
        search_roots.append(full_path)
        image_root = find_cirr_image_root(search_roots)

    if image_root is None:
        raise FileNotFoundError("Da thu tai CIRR tu Kaggle nhung khong tim thay thu muc goc co ca dev/ va test1/. Hay kiem tra noi dung dataset tai ve.")

    print(f"Thu muc goc chua cac split: {image_root}",flush=True)
    print("\nChuan bi annotation CIRR...", flush=True)

    annotation_root = find_cirr_annotation_root(search_roots)

    if annotation_root is not None:
        for folder_name in ["captions","captions_ext", "image_splits",]:
            source = annotation_root / folder_name
            destination = CIRR_DIR / folder_name

            if source.is_dir():
                copy_merge_directory(source, destination)
                print(f"Da sao chep {folder_name}: {destination}",flush=True)
    else:
        print("Khong tim thay annotation trong phan du lieu Kaggle. Thu dung ham prepare_cirr_metadata() hien co.",flush=True)

    required_files = [
        CIRR_DIR / "captions" / "cap.rc2.val.json",
        CIRR_DIR / "image_splits" / "split.rc2.val.json",
    ]

    if any(not path.is_file() or path.stat().st_size == 0 for path in required_files):
        prepare_cirr_metadata()

    # Kiem tra lai annotation bat buoc
    for path in required_files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Annotation CIRR chua day du: {path}")

    print("\nSao chep anh CIRR...", flush=True)

    for split in ["train", "dev", "test1"]:
        source = image_root / split
        destination = CIRR_DIR / split

        if not source.is_dir():
            if split == "train":
                print("Canh bao: dataset khong co train/. Se bo qua split train.",flush=True)
                continue

            raise FileNotFoundError(f"Thieu split anh bat buoc: {source}")

        copy_merge_directory(source, destination)
        image_count = count_cirr_images(destination)
        print(f"{split}: {image_count} anh -> {destination}",flush=True)

        if image_count == 0:
            raise RuntimeError(f"Split {split} khong co anh hop le.")

    print("\nKiem tra annotation...", flush=True)
    for path in required_files:
        print(f"{path}", flush=True)
    print("\nHoan tat chuan bi CIRR.", flush=True)
    print(f"Thu muc du lieu: {CIRR_DIR}", flush=True)
    print(f"Train: {count_cirr_images(CIRR_DIR / 'train')} anh",flush=True)
    print(f"Dev: {count_cirr_images(CIRR_DIR / 'dev')} anh",flush=True)
    print(f"Test1: {count_cirr_images(CIRR_DIR / 'test1')} anh",flush=True)

#Kiem tra xem du du lieu chua
def validate_structure():
    required_files = [
        ROOT / "cc" / "Train_GCC-training_output.csv",
        ROOT / "cc" / "Validation_GCC-1.1.0-Validation_output.csv",

        COCO_DIR / "annotations" / "instances_val2017.json",
        COCO_DIR / "coco_eval.csv",

        IMGNET_DIR / "imgnet_real_query.txt",
        IMGNET_DIR / "imgnet_targets.txt",

        CIRR_DIR / "captions" / "cap.rc2.val.json",
        CIRR_DIR / "image_splits" / "split.rc2.val.json",

        FASHION_DIR / "json" / "cap.dress.val.json",
        FASHION_DIR / "json" / "cap.shirt.val.json",
        FASHION_DIR / "json" / "cap.toptee.val.json",

        FASHION_DIR / "image_splits" / "split.dress.val.json",
        FASHION_DIR / "image_splits" / "split.shirt.val.json",
        FASHION_DIR / "image_splits" / "split.toptee.val.json",
    ]

    image_dirs = [
        ROOT / "cc_data" / "train",
        ROOT / "cc_data" / "val",
        COCO_DIR / "val2017",
        COCO_DIR / "val2017_masked",
        IMGNET_DIR / "real",
        IMGNET_DIR / "imagenet-r",
        CIRR_DIR / "dev",
        FASHION_DIR / "images",
    ]

    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    print("\nKiểm tra file")

    for path in required_files:
        valid = path.is_file()and path.stat().st_size > 0
        status = "OK" if valid else "MISSING/EMPTY"
        try:
            label = path.relative_to(ROOT)
        except ValueError:
            label = path

        print(f"[{status}] {label}")

    print("\nKiểm tra thư mục")

    for folder in image_dirs:
        found_image = False

        if folder.is_dir():
            for path in folder.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower() in image_extensions
                ):
                    found_image = True
                    break

        status = "HAS IMAGES" if found_image else "EMPTY/MISSING"

        try:
            label = folder.relative_to(ROOT)
        except ValueError:
            label = folder

        print(f"[{status}] {label}")

    print("\nLưu ý:")
    print("- Kiểm tra này xác nhận có file và ảnh,")
    print("  không khẳng định dataset đã tải đủ số lượng.")
    print("- Fashion-IQ của Pic2Word cần các ảnh .png.")
    print("- ImageNet validation gốc và ảnh CIRR cần được")
    print("  cung cấp theo quy trình truy cập tương ứng.")

def extract_archive(archive, dataset):
    destinations = {
        "cirr": CIRR_DIR,
        "fashioniq": FASHION_DIR,
        "imagenet-real": IMGNET_DIR / "real"
    }

    destination = destinations[dataset]
    archive = Path(archive)

    if not archive.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {archive}")

    if zipfile.is_zipfile(archive):
        extract_zip(archive,destination)

    elif tarfile.is_tarfile(archive):
        safe_extract_tar(archive,destination)

    else:
        raise ValueError("Không phải file ZIP, TAR hoặc TAR.GZ.")

    print(f"Đã giải nén {archive} vào {destination}")

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Pic2Word")
    parser.add_argument("--init", action="store_true", help="Tao cac thu muc dataset")
    parser.add_argument("--coco", action="store_true", help="Tai va chuan bi COCO validation")
    parser.add_argument("--imagenet-r", action="store_true", help="Tai va chuan bi ImageNet-R")
    parser.add_argument("--cirr-metadata", action="store_true", help="Lay annotation CIRR")
    parser.add_argument("--cirr-raw-archive",type=Path,help="Đường dẫn đến gói ảnh CIRR/NLVR2 đã được cấp quyền tải")
    parser.add_argument("--fashioniq-metadata", action="store_true", help="Lay annotation Fashion-IQ")
    parser.add_argument("--validate", action="store_true", help="Kiem tra dataset hien co")
    parser.add_argument("--cc3m-kaggle", action="store_true", help="Tai hai TSV CC3M tu KaggleHub")
    parser.add_argument("--cc3m", action="store_true", help="Tai anh tu TSV CC3M va tao CSV Pic2Word")
    parser.add_argument("--limit", type=int, default=0, help="So dong toi da moi split; 0 = tat ca")
    parser.add_argument("--workers", type=int, default=4, help="So luong tai anh dong thoi")
    parser.add_argument("--extract-archive",type=Path,help="Đường dẫn file ZIP/TAR cần giải nén")
    parser.add_argument("--fashioniq",action="store_true",help="Tải annotation và ảnh Fashion-IQ")
    parser.add_argument("--fashioniq-images",action="store_true",help="Tải ảnh Fashion-IQ")
    parser.add_argument("--fashion-limit",type=int,default=0,help="Giới hạn số ảnh Fashion-IQ; 0 = tất cả")
    parser.add_argument("--archive-dataset",choices=["cirr", "fashioniq", "imagenet-real"],help="Bộ dữ liệu đích khi giải nén file ZIP/TAR")
    parser.add_argument("--cirr-kaggle",action="store_true",help="Tai va chuan bi CIRR tu Kaggle")
    args = parser.parse_args()
    create_directories()

    if args.init:
        print(" Khoi tao thu muc hoan tat.", flush=True)

    if args.cirr_kaggle:
        prepare_cirr_kaggle()

    if args.cc3m_kaggle:
        download_cc3m_kaggle()

    if args.cc3m:
        prepare_cc3m(limit=args.limit, workers=args.workers)

    if args.coco:
        prepare_coco()

    if args.imagenet_r:
        prepare_imagenet_r()

    if args.cirr_metadata:
        prepare_cirr_metadata()

    if args.cirr_raw_archive:
        prepare_cirr_raw_images(args.cirr_raw_archive)

    if args.fashioniq_metadata:
        prepare_fashioniq_metadata()

    if args.validate:
        validate_structure()

    if args.extract_archive:
        if not args.archive_dataset:
            parser.error("--extract-archive cần có --archive-dataset")
        extract_archive(args.extract_archive,args.archive_dataset)

    if args.fashioniq:
        prepare_fashioniq_metadata()
        prepare_fashioniq_images(limit=args.fashion_limit, workers=args.workers)

    if args.fashioniq_images:
        prepare_fashioniq_images(limit=args.fashion_limit,workers=args.workers)

    actions = [args.init,args.coco,args.imagenet_r,args.cirr_metadata,args.cirr_kaggle,args.fashioniq_metadata,args.fashioniq,args.fashioniq_images,args.validate,args.cc3m_kaggle,args.cc3m,args.extract_archive is not None,]

    if not any(actions):
        parser.print_help()

if __name__ == "__main__":
    main()
