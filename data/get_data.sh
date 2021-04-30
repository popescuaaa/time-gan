SOLAR=https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/solar_nips.tar.gz
ELECTRICITY=https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/electricity_nips.tar.gz
EXCHANGE_RATE=https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/exchange_rate_nips.tar.gz
TAXI=https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/taxi_30min.tar.gz
TRAFFIC=https://github.com/mbohlkeschneider/gluon-ts/raw/mv_release/datasets/traffic_nips.tar.gz

wget $SOLAR
wget $ELECTRICITY
wget $EXCHANGE_RATE
wget $TAXI
wget $TRAFFIC
echo 'OK'
tar -xvf solar_nips.tar.gz
tar -xvf electricity_nips.tar.gz
tar -xvf exchange_rate_nips.tar.gz
tar -xvf taxi_30min.tar.gz
tar -xvf traffic_nips.tar.gz
echo 'OK'