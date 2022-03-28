from models.fixedtime_agent import FixedtimeAgent
from models.maxpressure_agent import MaxPressureAgent
from models.max_queuelength_agent import MaxQLAgent
from models.frap_agent import FRAPAgent
from models.gat_agent import GATAgent
from models.presslight_one import PressLightAgentOne
from models.simple_dqn_one import SimpleDQNAgentOne
from models.attention_light_agent import AttentionLightAgent


DIC_AGENTS = {
    "Fixedtime": FixedtimeAgent,
    "MaxPressure": MaxPressureAgent,

    "PressLight": PressLightAgentOne,
    "GAT": GATAgent,
    "FRAP": FRAPAgent,

    "MaxQL": MaxQLAgent,
    "DQN": SimpleDQNAgentOne,
    "Attention": AttentionLightAgent
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {

    "LIST_MODEL": ["Fixedtime",  "MaxPressure", "MaxQL", "FRAP",
                   "PressLight", "DQN", "GAT", "Attention"],
    "LIST_MODEL_NEED_TO_UPDATE": ["PressLight", "FRAP", "GAT", "Attention", "DQN"],

    "FORGET_ROUND": 20,
    "RUN_COUNTS": 3600,
    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "MIN_ACTION_TIME": 15,
    "MEASURE_TIME": 15,

    "BINARY_PHASE_EXPANSION": True,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "LIST_STATE_FEATURE": [
        "cur_phase",
        "pressure",
        "adjacency_matrix"
    ],
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],

}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "PATIENCE": 10,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [15, 15, 15, 15]
}
