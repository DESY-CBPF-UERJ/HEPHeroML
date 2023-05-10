# ![HEPHeroML](logoml.svg)

**HEPHeroML - Machine Learning tool for the DESY-CBPF-UERJ collaboration**

General information
-----------

* This code is meant to be used in association with the HEPHero framework.

* The training setup is made at the beginning of the file **train.py**.

* The code reads the data stored inside the **<output_path>/datasets** folder created by the tool **grouper.py** of the HEPHero framework.

* The training results and files are stored in **< output_path >/datasets/< period >/ML/< signal_name >/**


Quick start
-----------

Inside your private area (NOT in the eos or dust area and NOT inside a CMSSW release), download the code.  
```bash
git clone https://github.com/DESY-CBPF-UERJ/HEPHeroML.git
```

Source the hepenv environment before work with the HEPHeroML:
```
hepenv
```

Enter in the HEPHeroML directory:  
```bash
cd HEPHeroML
```

Know how many models(jobs) the code is setted to train (information needed to submit jobs):  
```bash
python train.py -j -1
```

Train the model in the position **n** of the list for the signal **signal_name**:  
```bash
python train.py -j n -s signal_name
```
Ex.:
```bash
python train.py -j 2 -s Signal_1000_100
```

Submit condor jobs:  
1. Make **submit_jobs.sh** an executable:  
```bash
chmod +x submit_jobs.sh
```   
2. See all flavours available for the jobs:  
```bash
./submit_jobs.sh help
```  
3. Submit all the **N** jobs the code is setted to train:  
```bash
./submit_jobs.sh flavour N signal_name
```  

After the jobs have finished, evaluate the training results:
```bash
python evaluate.py -s selection_name -p period
```
Ex.:
```bash
python evaluate.py -s ML -p 17
```
period = APV_16, 16, 17, or 18


