## Setup

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh

conda env create -f environment.yml
echo "export PATH=\"/home/iapucp/anaconda3/bin:$PATH\"" >> ˜/.bashrc 

ENV PATH /root/miniconda3/bin:$PATH
````

