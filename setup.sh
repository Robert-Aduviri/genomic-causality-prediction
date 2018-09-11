wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
export PATH="/home/$USER/miniconda3/bin:$PATH"
echo "export PATH=\"/home/$USER/miniconda3/bin:\$PATH\"" >> ~/.bashrc 
rm Miniconda3-latest-Linux-x86_64.sh
conda env create -f environment.yml
