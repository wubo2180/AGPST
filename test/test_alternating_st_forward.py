import pytest
import torch
import torch.nn as nn
from basicts.mask.alternating_st import AlternatingSTModel

class TestAlternatingSTForward:
    @pytest.fixture
    def model_config(self):
        return {
            'num_nodes': 10,
            'in_steps': 12,
            'out_steps': 12,
            'input_dim': 1,
            'embed_dim': 64,
            'num_heads': 4,
            'temporal_depth_1': 2,
            'spatial_depth_1': 2,
            'temporal_depth_2': 2,
            'spatial_depth_2': 2,
            'fusion_type': 'gated',
            'dropout': 0.1,
            'use_denoising': True
        }

    @pytest.fixture
    def model(self, model_config):
        return AlternatingSTModel(**model_config)

    def test_forward_basic(self, model):
        """测试基础正向传播"""
        # (B, T, N, C) 格式输入
        input_data = torch.randn(2, 12, 10, 1)  # batch=2, seq=12, nodes=10, channels=1
        output = model(input_data)
        assert output.shape == (2, 12, 10, 1)  # 验证输出形状

    def test_forward_alternative_input_format(self, model):
        """测试(B, N, T, C)格式输入"""
        input_data = torch.randn(2, 10, 12, 1)
        output = model(input_data)
        assert output.shape == (2, 12, 10, 1)

    def test_forward_with_denoising(self, model):
        """测试去噪模块工作正常"""
        model.use_denoising = True
        input_data = torch.randn(2, 12, 10, 1)
        output = model(input_data)
        assert not torch.isnan(output).any()

    def test_forward_without_denoising(self, model_config):
        """测试关闭去噪模块的情况"""
        model_config['use_denoising'] = False
        model = AlternatingSTModel(**model_config)
        input_data = torch.randn(2, 12, 10, 1)
        output = model(input_data)
        assert output.shape == (2, 12, 10, 1)

    def test_forward_edge_cases(self, model):
        """测试边界情况"""
        # 最小批次
        input_data = torch.randn(1, 12, 10, 1)
        output = model(input_data)
        assert output.shape == (1, 12, 10, 1)

        # 最小序列长度
        input_data = torch.randn(2, 1, 10, 1)
        with pytest.raises(ValueError):
            model(input_data)  # 应该抛出异常因为in_steps=12

    def test_forward_invalid_input(self, model):
        """测试无效输入"""
        # 错误维度
        with pytest.raises(ValueError):
            model(torch.randn(2, 12, 10))  # 缺少通道维度

        # 错误形状
        with pytest.raises(ValueError):
            model(torch.randn(2, 10, 12))  # 3D输入且不符合任何格式

    def test_forward_different_batch_sizes(self, model):
        """测试不同批次大小"""
        for batch_size in [1, 2, 4, 8]:
            input_data = torch.randn(batch_size, 12, 10, 1)
            output = model(input_data)
            assert output.shape == (batch_size, 12, 10, 1)

    def test_forward_different_node_counts(self, model_config):
        """测试不同节点数量"""
        for num_nodes in [5, 10, 20]:
            model_config['num_nodes'] = num_nodes
            model = AlternatingSTModel(**model_config)
            input_data = torch.randn(2, 12, num_nodes, 1)
            output = model(input_data)
            assert output.shape == (2, 12, num_nodes, 1)

    def test_forward_output_range(self, model):
        """测试输出范围合理性"""
        input_data = torch.randn(2, 12, 10, 1)
        output = model(input_data)
        assert output.min() > -10 and output.max() < 10  # 合理范围检查

    def test_forward_gradient_flow(self, model):
        """测试梯度能否正常回传"""
        input_data = torch.randn(2, 12, 10, 1, requires_grad=True)
        output = model(input_data)
        loss = output.mean()
        loss.backward()
        assert input_data.grad is not None

if __name__ == "__main__":
    pytest.main(["-v", __file__])