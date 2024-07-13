
import contextlib
import re
import shutil
import subprocess
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from urllib import parse, request

import requests
import torch

from wddetr.utils import LOGGER, TQDM, checks, clean_url, emojis, is_online, url2file

# Define Ultralytics GitHub assets maintained at https://github.com/ultralytics/assets
GITHUB_ASSETS_REPO = 'ultralytics/assets'
GITHUB_ASSETS_NAMES = [f'yolov8{k}{suffix}.pt' for k in 'nsmlx' for suffix in ('', '-cls', '-seg', '-pose')] + \
                      [f'yolov5{k}{resolution}u.pt' for k in 'nsmlx' for resolution in ('', '6')] + \
                      [f'yolov3{k}u.pt' for k in ('', '-spp', '-tiny')] + \
                      [f'yolo_nas_{k}.pt' for k in 'sml'] + \
                      [f'sam_{k}.pt' for k in 'bl'] + \
                      [f'FastSAM-{k}.pt' for k in 'sx'] + \
                      [f'rtdetr-{k}.pt' for k in 'lx'] + \
                      ['mobile_sam.pt']
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]


def is_url(url, check=True):
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    return False


def delete_dsstore(path, files_to_delete=('.DS_Store', '__MACOSX')):

    # Delete Apple .DS_store files
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f'Deleting {file} files: {matches}')
        for f in matches:
            f.unlink()


def zip_directory(directory, compress=True, exclude=('.DS_Store', '__MACOSX'), progress=True):

    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Unzip with progress bar
    files_to_zip = [f for f in directory.rglob('*') if f.is_file() and all(x not in f.name for x in exclude)]
    zip_file = directory.with_suffix('.zip')
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, 'w', compression) as f:
        for file in TQDM(files_to_zip, desc=f'Zipping {directory} to {zip_file}...', unit='file', disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file  # return path to zip file


def unzip_file(file, path=None, exclude=('.DS_Store', '__MACOSX'), exist_ok=False, progress=True):

    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        if len(top_level_dirs) > 1 or not files[0].endswith('/'):  # zip has multiple files at top level
            path = extract_path = Path(path) / Path(file).stem  # i.e. ../datasets/coco8
        else:  # zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # i.e. ../datasets/coco8

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(f'WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.')
            return path

        for f in TQDM(files, desc=f'Unzipping {file} to {Path(path).resolve()}...', unit='file', disable=not progress):
            zipObj.extract(f, path=extract_path)

    return path  # return unzip dir


def check_disk_space(url='https://ultralytics.com/assets/coco128.zip', sf=1.5, hard=True):

    try:
        r = requests.head(url)  # response
        assert r.status_code < 400, f'URL error for {url}: {r.status_code} {r.reason}'  # check response
    except Exception:
        return True  # requests issue, default to True

    # Check file size
    gib = 1 << 30  # bytes per GiB
    data = int(r.headers.get('Content-Length', 0)) / gib  # file size (GB)
    total, used, free = (x / gib for x in shutil.disk_usage('/'))  # bytes
    if data * sf < free:
        return True  # sufficient space

    # Insufficient space
    text = (f'WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, '
            f'Please free {data * sf - free:.1f} GB additional disk space and try again.')
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link):

    file_id = link.split('/d/')[1].split('/view')[0]
    drive_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    filename = None

    # Start session
    with requests.Session() as session:
        response = session.get(drive_url, stream=True)
        if 'quota exceeded' in str(response.content.lower()):
            raise ConnectionError(
                emojis(f'❌  Google Drive file download quota exceeded. '
                       f'Please try again later or download this file manually at {link}.'))
        for k, v in response.cookies.items():
            if k.startswith('download_warning'):
                drive_url += f'&confirm={v}'  # v is token
        cd = response.headers.get('content-disposition')
        if cd:
            filename = re.findall('filename="(.+)"', cd)[0]
    return drive_url, filename


def safe_download(url,
                  file=None,
                  dir=None,
                  unzip=True,
                  delete=False,
                  curl=False,
                  retry=3,
                  min_bytes=1E0,
                  progress=True):


    # Check if the URL is a Google Drive link
    gdrive = url.startswith('https://drive.google.com/')
    if gdrive:
        url, file = get_google_drive_file_info(url)

    f = dir / (file if gdrive else url2file(url)) if dir else Path(file)  # URL converted to filename
    if '://' not in str(url) and Path(url).is_file():  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        assert dir or file, 'dir or file required for download'
        desc = f"Downloading {url if gdrive else clean_url(url)} to '{f}'"
        LOGGER.info(f'{desc}...')
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        check_disk_space(url)
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = 'sS' * (not progress)  # silent
                    r = subprocess.run(['curl', '-#', f'-{s}L', url, '-o', f, '--retry', '3', '-C', '-']).returncode
                    assert r == 0, f'Curl return value {r}'
                else:  # urllib download
                    method = 'torch'
                    if method == 'torch':
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        with request.urlopen(url) as response, TQDM(total=int(response.getheader('Content-Length', 0)),
                                                                    desc=desc,
                                                                    disable=not progress,
                                                                    unit='B',
                                                                    unit_scale=True,
                                                                    unit_divisor=1024) as pbar:
                            with open(f, 'wb') as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(emojis(f'❌  Download failure for {url}. Environment is not online.')) from e
                elif i >= retry:
                    raise ConnectionError(emojis(f'❌  Download failure for {url}. Retry limit reached.')) from e
                LOGGER.warning(f'⚠️ Download failure, retrying {i + 1}/{retry} {url}...')

    if unzip and f.exists() and f.suffix in ('', '.zip', '.tar', '.gz'):
        from zipfile import is_zipfile

        unzip_dir = dir or f.parent  # unzip to dir if provided else unzip in place
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir, progress=progress)  # unzip
        elif f.suffix in ('.tar', '.gz'):
            LOGGER.info(f'Unzipping {f} to {unzip_dir.resolve()}...')
            subprocess.run(['tar', 'xf' if f.suffix == '.tar' else 'xfz', f, '--directory', unzip_dir], check=True)
        if delete:
            f.unlink()  # remove zip
        return unzip_dir


def get_github_assets(repo='ultralytics/assets', version='latest', retry=False):

    if version != 'latest':
        version = f'tags/{version}'  # i.e. tags/v6.2
    url = f'https://api.github.com/repos/{repo}/releases/{version}'
    r = requests.get(url)  # github api
    if r.status_code != 200 and r.reason != 'rate limit exceeded' and retry:  # failed and not 403 rate limit exceeded
        r = requests.get(url)  # try again
    if r.status_code != 200:
        LOGGER.warning(f'⚠️ GitHub assets check failure for {url}: {r.status_code} {r.reason}')
        return '', []
    data = r.json()
    return data['tag_name'], [x['name'] for x in data['assets']]  # tag, assets


def attempt_download_asset(file, repo='ultralytics/assets', release='v0.0.0'):

    from wddetr.utils import SETTINGS  # scoped for circular import

    # YOLOv3/5u updates
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ''))
    if file.exists():
        return str(file)
    elif (SETTINGS['weights_dir'] / file).exists():
        return str(SETTINGS['weights_dir'] / file)
    else:
        # URL specified
        name = Path(parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(('http:/', 'https:/')):  # download
            url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
            file = url2file(name)  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f'Found {clean_url(url)} locally at {file}')  # file already exists
            else:
                safe_download(url=url, file=file, min_bytes=1E5)

        elif repo == GITHUB_ASSETS_REPO and name in GITHUB_ASSETS_NAMES:
            safe_download(url=f'https://github.com/{repo}/releases/download/{release}/{name}', file=file, min_bytes=1E5)

        else:
            tag, assets = get_github_assets(repo, release)
            if not assets:
                tag, assets = get_github_assets(repo)  # latest release
            if name in assets:
                safe_download(url=f'https://github.com/{repo}/releases/download/{tag}/{name}', file=file, min_bytes=1E5)

        return str(file)


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(
                lambda x: safe_download(
                    url=x[0], dir=x[1], unzip=unzip, delete=delete, curl=curl, retry=retry, progress=threads <= 1),
                zip(url, repeat(dir)))
            pool.close()
            pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            safe_download(url=u, dir=dir, unzip=unzip, delete=delete, curl=curl, retry=retry)
