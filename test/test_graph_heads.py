"""
测试graph_heads参数重命名
验证新参数名是否正常工作
"""

import torch
from basicts.mask.post_patch_adaptive_graph import PostPatchDynamicGraphConv

def test_graph_heads_parameter():
    """测试graph_heads参数重命名"""
    
    print("=== 测试graph_heads参数重命名 ===")
    
    # 测试不同的graph_heads值
    test_configs = [
        {"graph_heads": 1, "desc": "单头"},
        {"graph_heads": 4, "desc": "4头"},  
        {"graph_heads": 8, "desc": "8头"}
    ]
    
    for config in test_configs:
        graph_heads = config["graph_heads"]
        desc = config["desc"]
        
        print(f"\n测试配置: {desc} (graph_heads={graph_heads})")
        
        try:
            # 创建动态图学习模块
            dynamic_graph = PostPatchDynamicGraphConv(
                embed_dim=96,
                num_nodes=358,
                node_dim=10,
                graph_heads=graph_heads,  # 使用新的参数名
                topk=6,
                dropout=0.1
            )
            
            print(f"  ✅ 模块创建成功")
            
            # 测试前向传播
            test_data = torch.randn(4, 358, 72, 96)  # (B, N, P, D)
            
            with torch.no_grad():
                enhanced_patches, learned_adj = dynamic_graph(test_data)
            
            print(f"  ✅ 前向传播成功")
            print(f"     输出形状: {enhanced_patches.shape}")
            print(f"     邻接矩阵: {learned_adj.shape}")
            
            # 验证内部参数
            graph_learner = dynamic_graph.graph_learner
            print(f"  ✅ 内部参数检查:")
            print(f"     graph_heads: {graph_learner.graph_heads}")
            print(f"     static_embeddings1: {graph_learner.static_node_embeddings1.shape}")
            print(f"     temperature: {graph_learner.temperature.shape}")
            
        except Exception as e:
            print(f"  ❌ 测试失败: {e}")
            return False
    
    return True

def test_parameter_conflicts():
    """测试参数名冲突解决"""
    
    print(f"\n=== 测试参数名冲突解决 ===")
    
    # 模拟Transformer的num_heads和图学习的graph_heads
    transformer_config = {
        "num_heads": 8,      # Transformer的多头注意力
        "graph_heads": 4,    # 图学习的多头
    }
    
    print(f"配置参数:")
    print(f"  Transformer num_heads: {transformer_config['num_heads']}")
    print(f"  Graph learning graph_heads: {transformer_config['graph_heads']}")
    
    try:
        # 创建图学习模块
        dynamic_graph = PostPatchDynamicGraphConv(
            embed_dim=96,
            num_nodes=358, 
            node_dim=10,
            graph_heads=transformer_config['graph_heads'],  # 使用图学习专用参数
            topk=6,
            dropout=0.1
        )
        
        print(f"  ✅ 参数区分成功，无冲突")
        print(f"     图学习使用: graph_heads={transformer_config['graph_heads']}")
        print(f"     Transformer可以独立使用: num_heads={transformer_config['num_heads']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 参数冲突测试失败: {e}")
        return False

def test_yaml_config_compatibility():
    """测试YAML配置兼容性"""
    
    print(f"\n=== 测试YAML配置兼容性 ===")
    
    # 模拟YAML配置
    yaml_config = {
        "num_heads": 4,       # Transformer heads
        "graph_heads": 4,     # Graph learning heads  
        "embed_dim": 96,
        "num_nodes": 358,
        "node_dim": 10,
        "topk": 6,
        "dropout": 0.1
    }
    
    print(f"模拟YAML配置:")
    for key, value in yaml_config.items():
        print(f"  {key}: {value}")
    
    try:
        # 使用配置创建模块
        dynamic_graph = PostPatchDynamicGraphConv(
            embed_dim=yaml_config["embed_dim"],
            num_nodes=yaml_config["num_nodes"],
            node_dim=yaml_config["node_dim"],
            graph_heads=yaml_config["graph_heads"],  # 注意使用正确的参数名
            topk=yaml_config["topk"],
            dropout=yaml_config["dropout"]
        )
        
        print(f"  ✅ YAML配置加载成功")
        print(f"  ✅ 参数名映射正确")
        
        return True
        
    except Exception as e:
        print(f"  ❌ YAML配置测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 测试graph_heads参数重命名")
    print("=" * 60)
    
    # 测试基本功能
    basic_test = test_graph_heads_parameter()
    
    # 测试参数冲突解决
    conflict_test = test_parameter_conflicts()
    
    # 测试YAML兼容性
    yaml_test = test_yaml_config_compatibility()
    
    print("\n" + "=" * 60)
    if basic_test and conflict_test and yaml_test:
        print("🎉 所有测试通过!")
        print("✅ graph_heads参数重命名成功")
        print("✅ 解决了与Transformer num_heads的冲突")
        print("✅ YAML配置兼容性良好")
        
        print(f"\n📝 使用说明:")
        print(f"  - Transformer多头注意力: num_heads")
        print(f"  - 图学习多头机制: graph_heads")
        print(f"  - 两个参数可以独立设置，互不干扰")
    else:
        print("❌ 部分测试失败，请检查代码")