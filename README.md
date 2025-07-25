# A Digital Twin Framework for Urban Parking Management and Mobility Forecasting
Framework Platform: https://www.labdma.unina.it/dtmob/home.html
## Abstract
Rapid urbanization and population growth have created significant challenges in urban mobility management, such as traffic congestion, inefficient public transportation, and environmental pollution. This paper presents A framework for urban parking management and mobility forecasting, development and implementation of a Digital Twin (DT) aimed at addressing issues within the context of smart mobility. The framework integrates a wide range of historical and real-time data, including parking meter transactions, revenue records, street occupancy rates, parking violations, and sensor-based parking slot utilization. Additionally, the data encompass weather conditions, temporal patterns (such as weekdays and peak hours), and agent shift schedules, offering a comprehensive dataset for analyzing and optimizing urban mobility dynamics. Descriptive statistics are used to identify key patterns, while advanced Machine Learning (ML) and Deep Learning (DL) algorithms enhance predictive and generative analytics, forecasting parking demand and simulating various mobility scenarios. These insights, combined with visualization tools, map data onto the urban landscape, enabling spatial planning and resource allocation. Moreover, the integration of Generative Artificial Intelligence (GenAI) models significantly improves the system's capabilities, generating realistic ``what-if'' scenarios that allow for virtual testing of mobility strategies before real-world implementation.
The results highlight the framework potential to improve urban mobility management, especially improving parking meter placement and enhancing the quality of urban mobility for users by reducing inefficiencies and improving accessibility. Tested on real-world data from the city of Caserta, the proposed framework has proven robust and adaptable, although expanding the dataset and refining specific components are necessary for fully realizing its potential and ensuring sustainable urban planning.


![Alt text](DT.png)


## Acknowledgments
This work was partially supported by the PNRR project FAIR -  Future AI Research (PE00000013), Spoke 3, under the NRRP MUR program funded by the NextGenerationEU.
This work was partially supported by PNRR Centro Nazionale HPC, Big Data e Quantum Computing, (CN 00000013) (CUP: E63C22000980007).
The authors would like to thank [K-city srl company](https://www.k-city.eu/) for their support, collaboration, and provision of the data. In particular, special thanks go to Dr. Giuseppe Morelli and Dr. Sebastiano Spina. The authors also thank the city of Caserta, including its mayor and staff, for their support.



## Data Availability
The full data used in this study are not publicly available due to confidentiality agreements and data protection policies. However, a subsample of such data with period April-May 2025 is provided for all the tools, with the exception of the agent shift calendar, whose data feeds have been anonymized.
Researchers interested in accessing the full data may contact the corresponding author to discuss potential access arrangements under appropriate confidentiality agreements.
The weather and air quality data, which are open source, can be downloaded from [Open-Meteo](https://open-meteo.com), while the Points of Interest data, obtained through [OpenStreetMap](https://www.openstreetmap.org).


## Installation

### 0. Prerequisites
All the code can be executed through [Docker Engine](https://docs.docker.com/engine/) with Rootless mode enabled. The reported installation instructions are valid with Ubuntu 22.04 LTS and similar.

For the execution of the scripts with a **CUDA-capable GPU**, we require the GPU and NVidia drivers compatible with **CUDA 12.1 or newer** and the installation of [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html). Furthermore, given the large size of the models for the training, a GPU with at least 24 GB of VRAM is required for this example data. For the complete data and the model of the case study we need at least 48 GB of VRAM.

### 1. Data download
The available data for the execution of all the scripts is provided at the following [link](https://drive.google.com/file/d/13JeMKgjJ-B-nm7pmzQIX7jiOQi71Lcar/view?usp=drive_link) (approximate size: 5 GB uncompressed).

Download the zip file in the `data/` folder and unzip it with the following command from the terminal:

```sh
unzip Data_DTMOB1.2.zip
```

this will create two folders inside data: `preprocessing` and `webapp`.

### 2. Build of the Docker Images
Run the following in a terminal at the main folder:

```sh
docker compose build --build-arg UID=$(id -u) --build-arg GID=$(id -g)
```

this will build all the docker images necessary for the execution of each step.

## Execution
### 1. Preprocessing
#### Starting the preprocessing container

Run the following command from the terminal:

```sh
docker compose run --rm preprocessing
```

this will start a new shell session in the generated container

#### 1.1. Forecasting

Three different configurations for the forecasting are available in the folder `code/preprocessing/configs/forecasting`:
- [Transactions](./code/preprocessing/configs/forecasting/transactions.yml), for the parking meter transactions forecasting
- [Revenue](./code/preprocessing/configs/forecasting/amount.yml), for the parking meter revenue forecasting
- [Road occupancy](./code/preprocessing/configs/forecasting/roads.yml), for the road occupancy forecasting

To run one of these configurations, run the following from inside the container:

```sh
bash run_forecasting.sh $CONFIG
```

where `$CONFIG` can be `transactions`, `amount`, or `roads` depending on the configuration you want to run.

The results are saved in the `results/preprocessing/forecasting` folder relative to the project folder.

#### 1.2. Generation
Three different configurations for the generation are available in the folder `code/preprocessing/configs/generation`:
- [First scenario](./code/preprocessing/configs/generation/1stscenario.yml), for the first scenario generation
- [Second scenario](./code/preprocessing/configs/generation/2ndscenario.yml), for the second scenario generation
- [Third scenario](./code/preprocessing/configs/generation/3rdscenario.yml), for the third scenario generation

To run one of these configurations, run the following from inside the container:

```sh
bash run_generation.sh $CONFIG
```

where `$CONFIG` can be `1stscenario`, `2ndscenario`, or `3rdscenario` depending on the configuration you want to run.

The results are saved in the `results/preprocessing/generation` folder.

#### 1.3. Scheduling
For the agent scheduling the only configurable option is the date in the format `YYYY-MM-DD`.

For example, to get the calendar for 12th January 2025, run the following:

```sh
bash run_scheduling.sh 2025-01-12
```

the calendar output is saved in `results/preprocessing/scheduling` folder.

### 2. Web application
#### Starting the web application container

Run the following command from the terminal:

```sh
docker compose up -d webapp nginx
```

This will start the web application and the nginx server in detached mode. All the logs from the web application are stored in the logs/webapp folder and the nginx logs in the logs/nginx folder.

#### Accessing the Web Interface
Open a web browser and navigate to: `http://localhost:8080`.


### 3. Interactive usage
A descriptive usage of the DTMOB framework is provided as a Jupyter Notebook [here](./code/example_usage/DTMOB_example.ipynb). 

#### Runnability
To make it runnable, you need to run the following command from the terminal:

```sh
docker compose up -d interactive
```

then open the Jupyter Notebook in your browser at `http://localhost:8081/notebooks/DTMOB_example.ipynb`. It is possible to change the arguments in the notebook to show different configurations of the framework.