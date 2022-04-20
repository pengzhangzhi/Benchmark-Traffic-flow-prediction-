# # # ST3DNet
cd ST3DNet
python prepareData.py
python main_taxiNYC.py
cd ..

# # 3D-CLoST
cd 3D-CLoST
python main_taxiNYC.py
cd ..

cd MST3D
python main_taxiNYC.py
cd ..

cd Autoencoder
python main_taxiNY_streednet.py
cd ..