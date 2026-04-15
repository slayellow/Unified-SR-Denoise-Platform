import yaml
import os
import subprocess
import copy

def read_yaml(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def write_yaml(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

def dict_to_yaml_str(data):
    return yaml.dump(data, default_flow_style=False, sort_keys=False)

def yaml_str_to_dict(yaml_str):
    try:
        return yaml.safe_load(yaml_str) or {}
    except Exception as e:
        return {"Error": f"Invalid YAML formatting: {e}"}

def stream_command(command, log_placeholder):
    """
    Execute a shell command and capture its stdout/stderr in real-time.
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        text=True,
        universal_newlines=True
    )

    full_log = ""
    for line in iter(process.stdout.readline, ""):
        if line:
            full_log += line
            display_log = "\n".join(full_log.split("\n")[-30:])
            log_placeholder.code(display_log, language='bash')
            
    process.stdout.close()
    return process.wait()

# ================================
# DOMAIN KNOWLEDGE (CONTEXT MAP)
# ================================
KNOWN_TASKS = ['sr', 'denoise', 'guide']
KNOWN_MODELS = [
    'quicksrnet_small', 'quicksrnet_medium', 'quicksrnet_large',
    'quicksrnet_denoise', 'quicksrnet_denoise_opt',
    'lrcsr', 'svsrnet', 'ddrnet', 'rrdbnet',
    'qcsawaresrnet_small', 'qcsawaresrnet_medium', 'qcsawaresrnet_large',
    'svfocussrnet', 'mambair', 'mambairv2',
    'corefusion', 'lapgsr', 'lapgsr_disc'
]
KNOWN_OPTIMIZERS = ['Adam', 'AdamW', 'SGD']
KNOWN_SCHEDULERS = ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']

HELP_TEXTS = {
    'batch_size': 'GPU 1대당 할당되는 배치 크기입니다. OOM 발생 시 줄이세요.',
    'patch_size': '학습 시 이미지를 크롭할 사이즈 지정 (일반적으로 64, 128, 256).',
    'gradient_accumulation_steps': '메모리가 부족할 때 실질 배치 사이즈를 뻥튀기하는 값입니다. (Total BS = batch_size * 이것 * GPU 수)',
    'epochs': '전체 학습 제한 반복 횟수입니다.',
    'lr': '옵티마이저의 Learning Rate 입니다.',
    'scale': 'Super Resolution(초해상화) 스케일 값. Denoise 단독은 보통 1입니다.',
    'upsampler': '해상도 복원 방식 (pixelshuffle 등)'
}

def render_dynamic_ui(st, data_dict, prefix="root"):
    """
    Context-Aware 스마트 파서가 적용된 UI 제너레이터입니다.
    """
    updated_dict = copy.deepcopy(data_dict)
    
    for key, value in data_dict.items():
        unique_key = f"{prefix}_{key}"
        label = str(key).replace('_', ' ').capitalize()
        help_txt = HELP_TEXTS.get(key, None)
        
        # 1. 특수 도메인 스니펫 (Context Mappings)
        if key == 'task' and isinstance(value, str):
            idx = KNOWN_TASKS.index(value) if value in KNOWN_TASKS else 0
            updated_dict[key] = st.selectbox("Task Mode", KNOWN_TASKS, index=idx, help=help_txt, key=unique_key)
            continue
            
        elif key == 'name' and 'model' in prefix:
            lower_val = str(value).lower()
            if lower_val in KNOWN_MODELS:
                idx = KNOWN_MODELS.index(lower_val)
            else:
                KNOWN_MODELS.insert(0, lower_val) # 만약 완전히 새로운 커스텀 모델일 때
                idx = 0
            updated_dict[key] = st.selectbox("Model Name", KNOWN_MODELS, index=idx, help=help_txt, key=unique_key)
            continue
            
        elif key == 'type' and 'optimizer' in prefix:
            idx = KNOWN_OPTIMIZERS.index(value) if value in KNOWN_OPTIMIZERS else 0
            updated_dict[key] = st.selectbox("Optimizer Type", KNOWN_OPTIMIZERS, index=idx, help=help_txt, key=unique_key)
            continue
            
        elif key == 'type' and 'scheduler' in prefix:
            idx = KNOWN_SCHEDULERS.index(value) if value in KNOWN_SCHEDULERS else 0
            updated_dict[key] = st.selectbox("Scheduler Type", KNOWN_SCHEDULERS, index=idx, help=help_txt, key=unique_key)
            continue
        
        # 2. Mutually Exclusive Loss 탐지 로직
        if key == 'loss' and isinstance(value, dict):
            st.markdown(f"### Loss Configuration (손실 함수)")
            
            # 진행 전, 현재 몇 개의 베이스 로스가 켜져있는지 스캔 (l1, mse, charbonnier 만 체크)
            active_bases = [lk for lk in ['l1', 'mse', 'charbonnier'] if value.get(lk, {}).get('enabled', False)]
            if len(active_bases) > 1:
                st.warning(f"⚠️ 주의: 여러 개의 기본 손실 함수({', '.join(active_bases)})가 동시에 활성화되어 있습니다! 충돌을 막기 위해 하나만 활성화(True)하는 것을 권장합니다.")
            
            with st.container():
                updated_dict[key] = render_dynamic_ui(st, value, unique_key)
            st.divider()
            continue

        # 3. 일반 동적 UI (Dictionary Recursion)
        if isinstance(value, dict):
            st.markdown(f"### {label}")
            with st.container():
                updated_dict[key] = render_dynamic_ui(st, value, unique_key)
            st.divider()
            
        # Boolean 값을 Toggle로 매핑
        elif isinstance(value, bool):
            updated_dict[key] = st.toggle(label, value=value, help=help_txt, key=unique_key)
            
        # 정수를 Slider로 매핑 (drag bar)
        elif isinstance(value, int):
            max_bound = max(1024, value * 4) if value > 0 else 100
            min_bound = 0 if value >= 0 else value * 4
            updated_dict[key] = st.slider(label, min_value=min_bound, max_value=max_bound, value=value, help=help_txt, key=unique_key)
            
        # Float를 Slider나 Text로 매핑
        elif isinstance(value, float):
            if value < 0.01:
                updated_dict[key] = st.number_input(label, value=value, format="%e", help=help_txt, key=unique_key)
            else:
                max_bound = max(10.0, float(value * 4.0))
                updated_dict[key] = st.slider(label, min_value=0.0, max_value=max_bound, value=value, help=help_txt, key=unique_key)
                
        # 리스트를 파싱 가능한 Text Input으로
        elif isinstance(value, list):
            list_str = ", ".join(map(str, value))
            new_str = st.text_input(label + " (Comma separated)", value=list_str, help=help_txt, key=unique_key)
            if new_str.strip():
                try:
                    if '.' in new_str:
                        updated_dict[key] = [float(x.strip()) for x in new_str.split(',')]
                    else:
                        updated_dict[key] = [int(x.strip()) if x.strip().lstrip('-').isdigit() else x.strip() for x in new_str.split(',')]
                except:
                    updated_dict[key] = [x.strip() for x in new_str.split(',')]
            else:
                updated_dict[key] = []
                
        # 나머지는 String
        else:
            updated_dict[key] = st.text_input(label, value=str(value) if value is not None else "", help=help_txt, key=unique_key)
    
    return updated_dict

# Default YAML templates (유지)
DEFAULT_TEMPLATES = {
    'train': {
        'task': 'sr',
        'model': {'name': 'quicksrnet_medium', 'scale': 2, 'in_chans': 3, 'img_size': 64},
        'train': {'epochs': 300, 'batch_size': 8, 'lr': 2e-4, 'gradient_accumulation_steps': 1, 'patch_size': 128},
        'loss': {'charbonnier': {'enabled': True, 'weight': 1.0}}
    },
    'data': {
        'dataset_type': 'sr',
        'crop_size': 256,
        'degradation': {'stage1': {'blur': {'enabled': True, 'prob': 0.5}}}
    },
    'finetune': {
        'task': 'sr',
        'model': {'name': 'svfocussrnet', 'scale': 2},
        'train': {'epochs': 50, 'batch_size': 32, 'lr': 5e-5, 'pretrained_path': 'path/to/best.pth'}
    },
    'aimet': {
        'quantization': {'param_bw': 8, 'output_bw': 8, 'scheme': 'tf_enhanced'},
        'pruning': {'target_ratio': 0.5, 'use_svd': True}
    }
}

def apply_apple_design(st):
    """
    Apple Aesthetic CSS.
    """
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", "Inter", sans-serif;
            font-weight: 400;
        }

        .block-container {
            padding-top: 3rem;
            padding-bottom: 3rem;
            max-width: 960px;
        }

        h1, h2, h3 {
            font-weight: 600 !important;
            letter-spacing: -0.015em !important;
        }

        div.stSlider > div[data-baseweb="slider"] div {
            border-radius: 9999px !important;
        }

        div.stButton > button:first-child {
            background-color: #000000;
            color: #ffffff;
            border-radius: 8px;
            font-weight: 500;
            border: none;
            padding: 0.5rem 1.2rem;
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #1d1d1f;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transform: translateY(-1px);
        }

        input:checked + div[data-baseweb="checkbox"] {
            background-color: #34c759 !important; 
        }

        input, select, textarea {
            border-radius: 6px !important;
            background-color: #fafafa !important;
            border: 1px solid #d2d2d7 !important;
            box-shadow: none !important;
        }
        input:focus, select:focus, textarea:focus {
            border-color: #000000 !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
