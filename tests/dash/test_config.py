import json
from malthusjax.dash.config import load_config

def test_load_config_basic(tmp_path):
    # tomllib reads from standard files. 
    # Let's write a simple toml string
    toml_str = '''
    [sources]
    test = "./data"
    
    [style]
    width = 15
    '''
    
    file_path = tmp_path / "config.toml"
    with open(file_path, "w") as f:
        f.write(toml_str)
        
    config = load_config(file_path)
    assert "sources" in config
    assert config["sources"]["test"] == "./data"
    assert config["style"]["width"] == 15

def test_load_config_includes(tmp_path):
    base_toml = '''
    [style]
    width = 10
    height = 5
    grid = true
    '''
    
    main_toml = '''
    includes = ["base.toml"]
    
    [style]
    width = 20
    '''
    
    base_path = tmp_path / "base.toml"
    with open(base_path, "w") as f:
        f.write(base_toml)
        
    main_path = tmp_path / "main.toml"
    with open(main_path, "w") as f:
        f.write(main_toml)
        
    config = load_config(main_path)
    
    # includes should be removed
    assert "includes" not in config
    
    # main overrides base width
    assert config["style"]["width"] == 20
    
    # base properties are inherited
    assert config["style"]["height"] == 5
    assert config["style"]["grid"] is True
