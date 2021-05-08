import wget
import tarfile

URLS = {
    "SOLAR": "https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/solar_nips.tar.gz",
    "ELECTRICITY": "https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/electricity_nips.tar.gz",
    "EXCHANGE_RATE": "https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/exchange_rate_nips.tar.gz",
    "TAXI": "https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/taxi_30min.tar.gz",
    "TRAFFIC": "https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/traffic_nips.tar.gz"
}

if __name__ == '__main__':
    for ds_name in URLS:
        archive_name = wget.download(URLS[ds_name])
        tar_file = tarfile.open(archive_name)
        tar_file.extract('.')
        tar_file.close()
