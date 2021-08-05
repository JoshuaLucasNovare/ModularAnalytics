# ModularAnalytics

## Setup
* Optional but recommended: Create virtualenv using: `python3 -m venv venv` or
* Create a virtual environment through conda using: `conda env create --name venv python=3.8.5`
1. Download, extract, and save [Apache Spark](https://www.apache.org/dyn/closer.lua/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz). For Windows OS: Needed [winutils](https://github.com/cdarlint/winutils). Download matching version.
2. Download and install Java version 8 or higher.
3. Set environment variable `SPARK_HOME` to full path where Apache Spark was saved.
4. Set environment variable `JAVA_HOME` to full path of Java bin folder.
5. Update VStudio Build Tools with Desktop Development with C++ with Windows 10 SDK and MSVC v142 - VS 2019 C++...
6. Install requirements using `pip install -r requirements.txt`
7. Use conda to install fbprophet: `conda install -c conda-forge fbprophet`


## Execution
1. Run `streamlit run main.py`.









