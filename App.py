import sys
import subprocess
import argparse
import os
from pathlib import Path

#Constants
RANDOM_STATE = 7374
DATA_SIZE = 5000

TEST_SIZE_PERCENTAGE = 0.3
MODEL_NAME = "RecruitmentModelTrained.pkl"

LIBS = [
    "numpy",
    "pandas",  
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "PySide6",
    "joblib",
    "tabulate"
]

LANGUAGES = ("English", "Spanish", "French", "German", "Chinese", "Japanese")
EDUCATION_LEVELS = ("High School", "Bachelor", "Master", "PhD")
GENDERS = ("Male", "Female")

P_LANGUAGES = (0.4, 0.3, 0.1, 0.1, 0.05, 0.05)
P_EDUCATION_LEVELS = (0.5, 0.3, 0.15, 0.05)
P_GENDERS = (0.7, 0.3)
P_HIRED = (0.6, 0.4)

EXPERIENCE_MIN = (0)
EXPERIENCE_MAX = (30)

AGE_MIN = 18
AGE_MAX = 50

CERTIFICATIONS_MIN = (0)
CERTIFICATIONS_MAX = (10)

# functions

def RunCommand(commandList: list[str], printCommand: bool = True, printError:bool=True) -> subprocess.CompletedProcess:
    print("⏳", " ".join(commandList))
    stdOutput = None if printCommand else subprocess.DEVNULL
    errorOutput = None if printError else subprocess.PIPE
    result = subprocess.run(commandList,stdout=stdOutput, stderr=errorOutput, text=True)
    if result.returncode != 0 and printError:
        print(result.stderr) 
    return result

def ShowEnvInfo():
    print("ℹ️  Environment Info:")
    print("Python Version:", sys.version)
    print("Platform:", sys.platform)
    print("Executable Path:", sys.executable)
    print("Current Working Directory:", os.getcwd())
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("sys.prefix:", sys.prefix)
    print("sys.base_prefix:", sys.base_prefix)

def InstallDeps():
    print("🟦 Installing deps.")
    RunCommand([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], printCommand=True) 
    RunCommand([sys.executable, "-m", "pip", "install", *LIBS], printCommand=True) 
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import seaborn as sns
    import matplotlib.pyplot as plt
    import joblib

def CreateVirtualEnv():
    print(f"🟦 Creating Virtual Environment.")
    venvPath = Path(".venv")
    if not venvPath.exists():
        print(f"Path: {venvPath.resolve()}")
        RunCommand([sys.executable, "-m", "venv", str(venvPath)])
    else:
        print("Virtual environment already exists at:", venvPath.resolve())
        
def ShowVirtualEnvInfo():
    ShowEnvInfo()
    print("🟦 Verifying installed deps.")
    for lib in LIBS:
        RunCommand([sys.executable, "-m", "pip", "show", lib], printCommand=False)

def ShowActivationCommand():
    path = Path('.venv/Scripts/activate').resolve()
    print(f"🟦 Command to activate the virtual environment. Copy the one you need to the clipboard.")
    print(f"🟡 Cmd\n    \"{path}\"")
    print(f"🟡 Windows PowerShell/PowerShell Core(Pwsh)\n   & \"{path}.ps1\"")
    print(f"🟡 Unix-like\n    source \"{path}\"")

def FitModel():
    print("ℹ️  Generating synthetic data for recruitment model.")
    syntheticData = {
        "Age": np.random.randint(AGE_MIN, AGE_MAX, DATA_SIZE),
        "Experience": np.random.randint(EXPERIENCE_MIN, EXPERIENCE_MAX, DATA_SIZE),
        "Gender": np.random.choice(GENDERS, DATA_SIZE, p=P_GENDERS),
        "EducationLevel": np.random.choice(EDUCATION_LEVELS, DATA_SIZE, p=P_EDUCATION_LEVELS),
        "Language": np.random.choice(LANGUAGES, DATA_SIZE, p=P_LANGUAGES),
        "Certifications": np.random.randint(CERTIFICATIONS_MIN,CERTIFICATIONS_MAX, DATA_SIZE),
        "Hired": np.random.choice([0, 1], DATA_SIZE, p=P_HIRED),
    }
    print("🟦 Encoding data.")
    dfMain = pd.DataFrame(syntheticData)
    educationLevelType = pd.CategoricalDtype(categories=EDUCATION_LEVELS, ordered=True)
    languageType = pd.CategoricalDtype(categories=LANGUAGES, ordered=True)
    genderType = pd.CategoricalDtype(categories=GENDERS, ordered=True)
    dfMain["EducationLevel"] = dfMain["EducationLevel"].astype(educationLevelType).cat.codes.astype("Int64")
    dfMain["Language"] = dfMain["Language"].astype(languageType).cat.codes.astype("Int64")
    dfMain["Gender"] = dfMain["Gender"].astype(genderType).cat.codes.astype("Int64")

    dfEncoded = dfMain.copy()
    x = dfEncoded.drop(columns=["Hired"])
    y = dfEncoded["Hired"]

    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=TEST_SIZE_PERCENTAGE, random_state=RANDOM_STATE
    )

    print("• Data shape, x(train and test) and y(train and test):")
    print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    print("ℹ️  Fitting the model.")
    model.fit(xTrain, yTrain)


    print("Model is Fitted.")
    joblib.dump(model, MODEL_NAME)
    print(f"Saved as '{MODEL_NAME}'.")

    print("ℹ️  Evaluating the model.")
    yPred = model.predict(xTest)

    print("🟦 Confusion Matrix")
    confusionMatrix = confusion_matrix(yTest, yPred, labels=[0, 1])
    labels = ["Not hired", "Hired"]  
    dfConfusionMatrix = pd.DataFrame(
        confusionMatrix,
        index=[f"Actual {l}" for l in labels],
        columns=[f"Predicted {l}" for l in labels],
    )
    print(dfConfusionMatrix)


    print("🟦 Accuracy score - Report")
    print(accuracy_score(yTest, yPred))

    print("🟦 Classification Report")
    print(classification_report(yTest, yPred))

    sns.heatmap(
        confusionMatrix,
        annot=True,
        fmt="d",
        cmap="inferno",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Recruitment Predicted")
    plt.ylabel("Recruitment Actual")
    plt.show()


# Main script
parser = argparse.ArgumentParser(description="Set up the project's virtual environment.")
parser.add_argument("--Install", action="store_true", help="Installs the dependencies in the active virtual environment.")
parser.add_argument("--Info", action="store_true", help="Shows environment information.")
parser.add_argument("--Create", action="store_true", help="Creates the virtual environment.")
parser.add_argument("--Activate", action="store_true", help="Gets the activation command for the environment.")
parser.add_argument("--Fit", action="store_true", help="Fits the model and save.")

args = parser.parse_args()


if args.Install:
    InstallDeps()
    sys.exit(0)

if args.Create:
    CreateVirtualEnv()
    sys.exit(0)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PySide6 import QtCore, QtWidgets, QtGui

if args.Info:
    ShowVirtualEnvInfo()
    sys.exit(0)

if args.Activate:
    ShowActivationCommand()
    sys.exit(0)

if args.Fit:
    FitModel()
    sys.exit(0)


def RunApp(): 
    print("🟦 Running App")
    app = QtWidgets.QApplication([])
    window = MyWindow()
    window.setWindowTitle("Recruitment Model App - Test")
    window.resize(800, 600)
    window.show()
    return app.exec()

class MyWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Age
        self.labelAge = QtWidgets.QLabel(f"Age ({AGE_MIN}-{AGE_MAX}):")
        self.labelAge.setWordWrap(False) 
        self.labelAge.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.labelAge.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.spinBoxAge = QtWidgets.QSpinBox()
        self.spinBoxAge.setRange(AGE_MIN, AGE_MAX)          
        self.spinBoxAge.setValue(AGE_MIN)              
        self.spinBoxAge.setSuffix(" years")          
        self.spinBoxAge.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.spinBoxAge.valueChanged.connect(lambda v: print("Age int:", v))
        
        # Experience
        self.labelExperience = QtWidgets.QLabel(f"Experience ({EXPERIENCE_MIN}-{EXPERIENCE_MAX}):")
        self.labelExperience.setWordWrap(False) 
        self.labelExperience.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.labelExperience.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.spinBoxExperience = QtWidgets.QSpinBox()
        self.spinBoxExperience.setRange(EXPERIENCE_MIN, EXPERIENCE_MAX)          
        self.spinBoxExperience.setValue(EXPERIENCE_MIN)              
        self.spinBoxExperience.setSuffix(" years")          
        self.spinBoxExperience.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.spinBoxExperience.valueChanged.connect(lambda v: print("Experience int:", v))

        # Gender
        self.labelGender = QtWidgets.QLabel("Gender:")
        self.labelGender.setWordWrap(False) 
        self.labelGender.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.labelGender.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.comboBoxGender = QtWidgets.QComboBox()
        # self.comboBoxGender.addItem("— gender —")
        # self.comboBoxGender.model().item(0).setEnabled(False)
        self.comboBoxGender.addItems(GENDERS)

        # Education Level
        self.labelEducationLevel = QtWidgets.QLabel("Education Level:")
        self.labelEducationLevel.setWordWrap(False) 
        self.labelEducationLevel.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.labelEducationLevel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.comboBoxEducactionLevel = QtWidgets.QComboBox()
        # self.comboBoxEducactionLevel.addItem("— level —")
        # self.comboBoxEducactionLevel.model().item(0).setEnabled(False)
        self.comboBoxEducactionLevel.addItems(EDUCATION_LEVELS)

        # Language
        self.labelLanguage = QtWidgets.QLabel("Language:")
        self.labelLanguage.setWordWrap(False) 
        self.labelLanguage.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.labelLanguage.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.comboBoxLanguage = QtWidgets.QComboBox()
        # self.comboBoxLanguage.addItem("— level —")
        # self.comboBoxLanguage.model().item(0).setEnabled(False)
        self.comboBoxLanguage.addItems(LANGUAGES)

        # Certifications
        self.labelCertifications = QtWidgets.QLabel(f"Certifications ({CERTIFICATIONS_MIN}-{CERTIFICATIONS_MAX}):")
        self.labelCertifications.setWordWrap(False) 
        self.labelCertifications.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.labelCertifications.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.spinBoxCertifications = QtWidgets.QSpinBox()
        self.spinBoxCertifications.setRange(CERTIFICATIONS_MIN, CERTIFICATIONS_MAX)          
        self.spinBoxCertifications.setValue(CERTIFICATIONS_MIN)              
        self.spinBoxCertifications.setSuffix(" certifications")          
        self.spinBoxCertifications.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        self.spinBoxCertifications.valueChanged.connect(lambda v: print("Certifications int:", v))

        # Button and Text
        self.label = QtWidgets.QLabel(f"Click the button to predict hiring status ℹ️.")
        self.label.setWordWrap(False) 
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter) 
        self.label.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred,
            QtWidgets.QSizePolicy.Fixed       
        )
        self.text = QtWidgets.QLabel("Click the button to predict hiring status!!!", alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.buttonFit = QtWidgets.QPushButton("FIT MODEL")
        self.buttonPredict = QtWidgets.QPushButton("PREDICT")

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(20)
        self.layout.addWidget(self.labelAge)
        self.layout.addWidget(self.spinBoxAge)
        self.layout.insertItem(0, QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed))
        self.layout.addWidget(self.labelExperience)
        self.layout.addWidget(self.spinBoxExperience)
        self.layout.insertItem(0, QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed))
        self.layout.addWidget(self.labelGender)
        self.layout.addWidget(self.comboBoxGender)
        self.layout.insertItem(0, QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed))
        self.layout.addWidget(self.labelEducationLevel)
        self.layout.addWidget(self.comboBoxEducactionLevel)
        self.layout.insertItem(0, QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed))
        self.layout.addWidget(self.labelLanguage)
        self.layout.addWidget(self.comboBoxLanguage)
        self.layout.insertItem(0, QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed))
        self.layout.addWidget(self.labelCertifications)
        self.layout.addWidget(self.spinBoxCertifications)
        self.layout.insertItem(0, QtWidgets.QSpacerItem(0, 10, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed))
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.buttonFit)
        self.layout.addWidget(self.buttonPredict)

        self.buttonPredict.clicked.connect(self.buttonPredictClick)
        self.buttonFit.clicked.connect(self.buttonFitClick)

    @QtCore.Slot()
    def buttonFitClick(self):
        FitModel()
        QtWidgets.QMessageBox.information(self, "Info", "Model fitted and saved.")

    @QtCore.Slot()
    def buttonPredictClick(self):
        try:
            model = joblib.load(MODEL_NAME)
            if not model:
                QtWidgets.QMessageBox.critical(self, "Error", "Model is not loaded.")
                return
            inputData =pd.DataFrame({
                'Age': [self.spinBoxAge.value()],
                'Experience': [self.spinBoxExperience.value()],
                "Gender": [self.comboBoxGender.currentIndex() -1],
                "EducationLevel": [self.comboBoxEducactionLevel.currentIndex() - 1],  
                "Language": [self.comboBoxLanguage.currentIndex() - 1],  
                "Certifications": [self.spinBoxCertifications.value()],
            })
            prediction = model.predict(inputData)[0]
            probs = model.predict_proba(inputData)[0] 
            classes = model.classes_            
            print("ℹ️  Probabilities:")
            for idx, c in enumerate(classes):
                print(f"Class: {c}, Probability: {probs[idx]:.4f}")

            map_idx = {c: i for i, c in enumerate(classes)}
            p_pred = probs[map_idx[prediction]]
            classesText = ["Not Hired", "Hired"]
            selectedText = f"Age: {self.spinBoxAge.value()}, Experience: {self.spinBoxExperience.value()}, Gender: {self.comboBoxGender.currentText()}, Education Level: {self.comboBoxEducactionLevel.currentText()}, Language: {self.comboBoxLanguage.currentText()}, Certifications: {self.spinBoxCertifications.value()}"
            QtWidgets.QMessageBox.information(self, "Prediction Result", f"Prediction: {classesText[prediction]} | Probability(Prediction): {p_pred:.4f}\n\n{selectedText}")

        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self, "Error", f"Model file '{MODEL_NAME}' not found.")
            return


sys.exit(RunApp())