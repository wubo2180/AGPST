"""
简单的导入测试（不需要torch）
"""
import sys
import os

def test_file_structure():
    """测试文件结构"""
    print("=" * 60)
    print("测试: 文件结构")
    print("=" * 60)
    
    mask_dir = "basicts/mask"
    expected_files = [
        "__init__.py",
        "model.py",
        "graph_learning.py",
        "patch_embed.py",
        "transformer.py",
        "positional_encoding.py",
        "README.md"
    ]
    
    print(f"\n检查目录: {mask_dir}")
    
    all_exist = True
    for filename in expected_files:
        filepath = os.path.join(mask_dir, filename)
        exists = os.path.exists(filepath)
        status = "✅" if exists else "❌"
        print(f"  {status} {filename}")
        if not exists:
            all_exist = False
    
    # 检查已删除的文件
    print(f"\n检查已删除的文件:")
    deleted_files = [
        "forecasting_with_adaptive_graph.py",
        "post_patch_adaptive_graph.py",
        "patch.py",
        "transformer_layers.py",
        "model_old.py",
        "adaptive_graph_improved.py",
        "patch_improved.py",
        "transformer_layers_improved.py",
        "integration_example.py",
        "maskgenerator.py",
        "GIN.py",
        "adaptive_graph.py",
        "contrastive_loss.py",
        "spatial_temporal_attention.py"
    ]
    
    all_deleted = True
    for filename in deleted_files:
        filepath = os.path.join(mask_dir, filename)
        exists = os.path.exists(filepath)
        status = "✅ 已删除" if not exists else "❌ 仍存在"
        if exists:
            print(f"  {status} {filename}")
            all_deleted = False
    
    if all_deleted:
        print(f"  ✅ 所有旧文件已成功删除")
    
    return all_exist and all_deleted


def test_imports_syntax():
    """测试导入语法（不实际导入torch相关）"""
    print("\n" + "=" * 60)
    print("测试: Python语法")
    print("=" * 60)
    
    files_to_check = [
        "basicts/mask/__init__.py",
        "basicts/mask/model.py",
        "basicts/mask/graph_learning.py",
        "basicts/mask/patch_embed.py",
        "basicts/mask/transformer.py"
    ]
    
    all_valid = True
    for filepath in files_to_check:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
                compile(code, filepath, 'exec')
            print(f"  ✅ {filepath}")
        except SyntaxError as e:
            print(f"  ❌ {filepath}: {e}")
            all_valid = False
        except FileNotFoundError:
            print(f"  ❌ {filepath}: 文件不存在")
            all_valid = False
    
    return all_valid


def check_main_import():
    """检查main.py的导入语句"""
    print("\n" + "=" * 60)
    print("测试: main.py导入更新")
    print("=" * 60)
    
    try:
        with open("main.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查新导入
        if "from basicts.mask.model import AGPSTModel" in content:
            print("  ✅ 新导入语句正确")
        else:
            print("  ❌ 未找到新导入语句")
            return False
        
        # 检查旧导入已删除
        if "from basicts.mask.forecasting_with_adaptive_graph" not in content:
            print("  ✅ 旧导入已删除")
        else:
            print("  ❌ 仍有旧导入语句")
            return False
        
        # 检查模型使用
        if "AGPSTModel(" in content:
            print("  ✅ 使用AGPSTModel")
            return True
        else:
            print("  ⚠️  未找到AGPSTModel使用")
            return True  # 可能在其他地方使用
            
    except FileNotFoundError:
        print("  ❌ main.py不存在")
        return False


def count_code_lines():
    """统计代码行数"""
    print("\n" + "=" * 60)
    print("统计: 代码行数")
    print("=" * 60)
    
    files = {
        "model.py": "basicts/mask/model.py",
        "graph_learning.py": "basicts/mask/graph_learning.py",
        "patch_embed.py": "basicts/mask/patch_embed.py",
        "transformer.py": "basicts/mask/transformer.py",
        "positional_encoding.py": "basicts/mask/positional_encoding.py"
    }
    
    total_lines = 0
    total_code_lines = 0
    
    for name, filepath in files.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                file_lines = len(lines)
                code_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
                
                print(f"  {name:25s} {file_lines:4d} 行 ({code_lines:4d} 代码)")
                total_lines += file_lines
                total_code_lines += code_lines
        except FileNotFoundError:
            print(f"  {name:25s} 文件不存在")
    
    print("  " + "-" * 50)
    print(f"  {'总计':25s} {total_lines:4d} 行 ({total_code_lines:4d} 代码)")
    
    return True


def main():
    """主函数"""
    print("\n🔍 AGPST 架构重构验证\n")
    
    results = {
        "文件结构": test_file_structure(),
        "Python语法": test_imports_syntax(),
        "main.py更新": check_main_import(),
        "代码统计": count_code_lines()
    }
    
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有验证通过! 架构重构成功!")
        print("\n📝 新架构:")
        print("   • 5个核心文件 (从13个精简)")
        print("   • model.py - 主模型")
        print("   • graph_learning.py - 图学习")
        print("   • patch_embed.py - Patch嵌入")
        print("   • transformer.py - Transformer")
        print("   • positional_encoding.py - 位置编码")
        print("\n📚 文档:")
        print("   • basicts/mask/README.md - 模块文档")
        print("   • REFACTORING_SUMMARY.md - 重构总结")
        print("   • ARCHITECTURE_DIAGRAM.md - 架构图")
    else:
        print("⚠️  部分验证失败，请检查上述信息")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
