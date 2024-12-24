from pathlib import Path

BERT_TYPE = './Bio_ClinicalBERT'
VIT_TYPE = './Swin_Transformer'

IMG_SIZE = 224
IMG_MEAN = .5862785803043838
IMG_STD = .27950088968644304
SELECT_NUM = 4
VIT_BERT_LEARNING_RATE = 2e-5
SELECT_MODEL_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 42
DATASET_PATH = "D:\\Codes\\ML\\fl-clip\\data\\data_set"
DATALIST_PATH = "D:\\Codes\\ML\\fl-clip\\data\\data_list"
# DATASET_PATH = "/home/ligong2/FL/fl-clip/data/data_set"
# DATALIST_PATH = "/home/ligong2/FL/fl-clip/data/data_list"
# DATASET_PATH = "/root/autodl-tmp/fl-clip/data/data_set"
# DATALIST_PATH = "/root/autodl-tmp/fl-clip/data/data_list"
CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4"]
ROUNDS = 100
CLIENTS_LABEL = {"client_1": [1, 0, 0, 0], "client_2": [0, 1, 0, 0], "client_3": [0, 0, 1, 0], "client_4": [0, 0, 0, 1]}
CLIENT_ACC = {"client_1": 0, "client_2": 0, "client_3": 0, "client_4": 0}
GLOBAL_ACC = 0
SELECT_ACC = 0
THRESHOLD = 0.4
CLIENTS_WEIGHT = {"client_1": 1/4, "client_2": 1/4, "client_3": 1/4, "client_4": 1/4}
LOGFILE = "./outputs/log/log.txt"
CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}

COVID_TASKS = [
    'Normal',
    'COVID',
]
COVID_CLASS_PROMPTS = {
    'COVID': {
        'adjective': ['patchy', 'confluent'],
        'description': ['ground glass'],
        'subtype': ['opacity', 'consolidation'],
        'location': ['in peripheral', 'in mid', 'in lower'],
    }
}

RSNA_TASKS = [
    'Normal',
    'Pneumonia',
]
RSNA_CLASS_PROMPTS = {
    'Pneumonia': {
        'adjective': ['round', 'early', 'focal', 'multifocal', 'small', ''],
        'subtype': ['bacterial', 'viral', 'mycoplasma', ''],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left middle lobe",
            "at the right middle lobe",
            ""
        ]
    },
    'Normal': {
        'adjective': ['clear', 'well-expanded', 'stable', 'unchanged', ''],
        'condition': ['no pleural effusion', 'no pneumothorax', 'no focal consolidation', 'no significant change', ''],
        'location': [
            'in the right lung field',
            'in the left lung field',
            'in the lower lobes',
            'in the upper lobes',
            'throughout the lung fields',
            ''
        ]
    }
}

WEIGHTS_NAME = 'pytorch_model.bin'

# store the URL of pretrained weights, `dev` needs to change to `main` after merging it to main branch.
PRETRAINED_URL_MEDCLIP_RESNET = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_resnet_weight.txt'
PRETRAINED_URL_MEDCLIP_VIT = 'https://github.com/RyanWangZf/MedCLIP/raw/main/medclip/medclip_vit_weight.txt'

DATA_BASE_DIR = Path('./data/data_set')
RSNA_DATA_DIR = DATA_BASE_DIR / "RSNA_Pneumonia"
RSNA_ORIGINAL_TRAIN_CSV = RSNA_DATA_DIR / "stage_2_train_labels.csv"
RSNA_CLASSINFO_CSV = RSNA_DATA_DIR / "stage_2_detailed_class_info.csv"
RSNA_TRAIN_CSV = RSNA_DATA_DIR / "train.csv"
RSNA_VALID_CSV = RSNA_DATA_DIR / "val.csv"
RSNA_TEST_CSV = RSNA_DATA_DIR / "test.csv"
RSNA_DETECTION_TRAIN_PKL = RSNA_DATA_DIR / "train.pkl"
RSNA_DETECTION_VALID_PKL = RSNA_DATA_DIR / "val.pkl"
RSNA_DETECTION_TEST_PKL = RSNA_DATA_DIR / "test.pkl"

local_bz = {
    "client_1": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "client_2": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "client_3": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "client_4": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
}
select_len = {
    'mlp': 5,
    'ca':  5,
}
RSNA_IMG_DIR = RSNA_DATA_DIR / "stage_2_train_images"
RSNA_TRAIN_PCT = 0.7

PNEUMOTHORAX_DATA_DIR = DATA_BASE_DIR / "SIIM_Pneumothorax"
PNEUMOTHORAX_ORIGINAL_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_TRAIN_CSV = PNEUMOTHORAX_DATA_DIR / "train.csv"
PNEUMOTHORAX_VALID_CSV = PNEUMOTHORAX_DATA_DIR / "valid.csv"
PNEUMOTHORAX_TEST_CSV = PNEUMOTHORAX_DATA_DIR / "test.csv"
PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_IMG_SIZE = 1024
PNEUMOTHORAX_TRAIN_PCT = 0.7

batch_size = {
    "prox" :{"client_1":10, "client_2": 10, "client_3":10, "client_4": 10},
    "moon": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "sm": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "fed": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "avg": {"client_1": 10, "client_2": 10, "client_3": 10, "client_4": 10},
    "pfm": {"client_1": 1, "client_2": 1, "client_3": 1, "client_4": 1},
    "mgca": {"client_1": 1, "client_2": 1, "client_3": 1, "client_4": 1},
}
total_size ={
    "prox": {'rsna': 500, 'covid': 500},
    "moon": {'rsna': 500, 'covid': 500},
    "sm": {'rsna': 500, 'covid': 500},
    "fed": {'rsna': 500, 'covid': 500},
    "avg": {'rsna': 500, 'covid': 500},
    "pms": {'rsna': 500, 'covid': 500},
    "mgca": {'rsna': 500, 'covid': 500},
}
