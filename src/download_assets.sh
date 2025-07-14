wget 'https://www.ipb.uni-bonn.de/html/projects/roggiolani2025pp/best.ckpt'
wget -O ./metrics/pointmlp_8k.pth 'https://www.ipb.uni-bonn.de/html/projects/roggiolani2025pp/pointmlp_8k.pth'
wget -O ./metrics/pointnet_on_single_view.pth 'https://www.ipb.uni-bonn.de/html/projects/roggiolani2025pp/pointnet_on_single_view.pth'
wget -O ./metrics/bbc.pt 'https://www.ipb.uni-bonn.de/html/projects/roggiolani2025pp/bbc.pt'
wget 'https://www.ipb.uni-bonn.de/html/projects/roggiolani2025pp/data.zip'
unzip data.zip
rm data.zip
