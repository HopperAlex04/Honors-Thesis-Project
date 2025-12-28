"""
Tests for the metrics graph generation script.

This test suite covers:
- Metrics file parsing
- Phase aggregation
- Metric extraction
- Statistical calculations
- Graph generation (basic validation)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np

# Try to import pytest and unittest
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import unittest
    HAS_UNITTEST = True
except ImportError:
    HAS_UNITTEST = False

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Mock matplotlib/seaborn before importing the module
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Create mock matplotlib
    class MockMatplotlib:
        class pyplot:
            @staticmethod
            def savefig(*args, **kwargs):
                pass
            @staticmethod
            def close(*args, **kwargs):
                pass
            @staticmethod
            def subplots(*args, **kwargs):
                return MagicMock(), MagicMock()
            @staticmethod
            def style():
                class use:
                    @staticmethod
                    def __call__(*args, **kwargs):
                        pass
                return use()
            @staticmethod
            def tight_layout(*args, **kwargs):
                pass
    
    sys.modules['matplotlib'] = MockMatplotlib()
    sys.modules['matplotlib.pyplot'] = MockMatplotlib.pyplot()

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    # Create mock seaborn
    class MockSeaborn:
        @staticmethod
        def set_style(*args, **kwargs):
            pass
        @staticmethod
        def color_palette(*args, **kwargs):
            return ['#000000', '#111111', '#222222']
    
    sys.modules['seaborn'] = MockSeaborn()

# Now import the module
from generate_metrics_graphs import (
    parse_metrics_file,
    extract_round_number,
    aggregate_phase_metrics,
    extract_metric_series,
    calculate_statistics,
    generate_graph
)


class TestParseMetricsFile(unittest.TestCase if not HAS_PYTEST else object):
    """Tests for parse_metrics_file function."""
    
    def test_parse_valid_metrics_file(self):
        """Test parsing a valid JSONL metrics file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            test_data = [
                {"episode": 0, "p1_win_rate": 0.5, "loss": 0.1},
                {"episode": 1, "p1_win_rate": 0.6, "loss": 0.09},
                {"episode": 2, "p1_win_rate": 0.7, "loss": 0.08}
            ]
            for entry in test_data:
                f.write(json.dumps(entry) + "\n")
            f.flush()
            file_path = Path(f.name)
        
        try:
            result = parse_metrics_file(file_path)
            assert len(result) == 3
            assert result[0]["episode"] == 0
            assert result[1]["p1_win_rate"] == 0.6
            assert result[2]["loss"] == 0.08
        finally:
            file_path.unlink()
    
    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            file_path = Path(f.name)
        
        try:
            result = parse_metrics_file(file_path)
            assert result == []
        finally:
            file_path.unlink()
    
    def test_parse_file_with_invalid_json(self):
        """Test parsing file with some invalid JSON lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"episode": 0, "p1_win_rate": 0.5}) + "\n")
            f.write("invalid json line\n")
            f.write(json.dumps({"episode": 1, "p1_win_rate": 0.6}) + "\n")
            f.flush()
            file_path = Path(f.name)
        
        try:
            result = parse_metrics_file(file_path)
            assert len(result) == 2
            assert result[0]["episode"] == 0
            assert result[1]["episode"] == 1
        finally:
            file_path.unlink()
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file."""
        file_path = Path("/nonexistent/path/metrics.jsonl")
        result = parse_metrics_file(file_path)
        assert result == []


class TestExtractRoundNumber(unittest.TestCase if not HAS_PYTEST else object):
    """Tests for extract_round_number function."""
    
    def test_extract_round_from_selfplay(self):
        """Test extracting round number from selfplay filename."""
        filename = "metrics_hand_only_round_5_selfplay.jsonl"
        assert extract_round_number(filename) == 5
    
    def test_extract_round_from_validation(self):
        """Test extracting round number from validation filename."""
        filename = "metrics_hand_only_round_10_vs_randomized_trainee_first.jsonl"
        assert extract_round_number(filename) == 10
    
    def test_extract_round_from_invalid_filename(self):
        """Test extracting round number from invalid filename."""
        filename = "invalid_filename.jsonl"
        assert extract_round_number(filename) is None
    
    def test_extract_round_with_multiple_digits(self):
        """Test extracting round number with multiple digits."""
        filename = "metrics_hand_only_round_123_selfplay.jsonl"
        assert extract_round_number(filename) == 123


class TestAggregatePhaseMetrics(unittest.TestCase if not HAS_PYTEST else object):
    """Tests for aggregate_phase_metrics function."""
    
    def test_aggregate_selfplay_metrics(self):
        """Test aggregating selfplay metrics across rounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            log_dir = base_dir / "action_logs" / "hand_only"
            log_dir.mkdir(parents=True)
            
            # Create test metrics files
            for round_num in [0, 1, 2]:
                file_path = log_dir / f"metrics_hand_only_round_{round_num}_selfplay.jsonl"
                with open(file_path, 'w') as f:
                    for episode in range(3):
                        data = {
                            "episode": episode,
                            "p1_win_rate": 0.5 + round_num * 0.1,
                            "loss": 0.1 - round_num * 0.01
                        }
                        f.write(json.dumps(data) + "\n")
            
            result = aggregate_phase_metrics("hand_only", "selfplay", base_dir)
            
            assert len(result) == 3
            assert 0 in result
            assert 1 in result
            assert 2 in result
            assert len(result[0]) == 3
            assert len(result[1]) == 3
            assert len(result[2]) == 3
    
    def test_aggregate_validation_metrics(self):
        """Test aggregating validation metrics with trainee_first and trainee_second."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            log_dir = base_dir / "action_logs" / "hand_only"
            log_dir.mkdir(parents=True)
            
            # Create test metrics files
            round_num = 0
            first_file = log_dir / f"metrics_hand_only_round_{round_num}_vs_randomized_trainee_first.jsonl"
            second_file = log_dir / f"metrics_hand_only_round_{round_num}_vs_randomized_trainee_second.jsonl"
            
            with open(first_file, 'w') as f:
                for episode in range(2):
                    data = {"episode": episode, "p1_win_rate": 0.6}
                    f.write(json.dumps(data) + "\n")
                f.flush()  # Ensure data is written
            
            with open(second_file, 'w') as f:
                for episode in range(2):
                    data = {"episode": episode, "p1_win_rate": 0.7}
                    f.write(json.dumps(data) + "\n")
                f.flush()  # Ensure data is written
            
            # Verify files were written correctly
            from generate_metrics_graphs import parse_metrics_file
            first_episodes = parse_metrics_file(first_file)
            second_episodes = parse_metrics_file(second_file)
            assert len(first_episodes) == 2, f"First file should have 2 episodes, got {len(first_episodes)}"
            assert len(second_episodes) == 2, f"Second file should have 2 episodes, got {len(second_episodes)}"
            
            result = aggregate_phase_metrics("hand_only", "vs_randomized", base_dir, combine_positions=True)
            
            # The function should find and combine the episodes
            assert len(result) == 1, (
                f"Expected 1 round, got {len(result)}: {list(result.keys())}. "
                f"First episodes: {len(first_episodes)}, Second episodes: {len(second_episodes)}. "
                f"Files exist: first={first_file.exists()}, second={second_file.exists()}"
            )
            assert round_num in result
            assert len(result[round_num]) == 4  # 2 from first + 2 from second
    
    def test_aggregate_with_rounds_filter(self):
        """Test aggregating with rounds filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            log_dir = base_dir / "action_logs" / "hand_only"
            log_dir.mkdir(parents=True)
            
            # Create test metrics files for rounds 0-4
            for round_num in range(5):
                file_path = log_dir / f"metrics_hand_only_round_{round_num}_selfplay.jsonl"
                with open(file_path, 'w') as f:
                    data = {"episode": 0, "p1_win_rate": 0.5}
                    f.write(json.dumps(data) + "\n")
            
            result = aggregate_phase_metrics("hand_only", "selfplay", base_dir, rounds_filter=[1, 3])
            
            assert len(result) == 2
            assert 1 in result
            assert 3 in result
            assert 0 not in result
            assert 2 not in result
            assert 4 not in result
    
    def test_aggregate_nonexistent_training_type(self):
        """Test aggregating for non-existent training type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            result = aggregate_phase_metrics("nonexistent", "selfplay", base_dir)
            assert result == {}


class TestExtractMetricSeries(unittest.TestCase if not HAS_PYTEST else object):
    """Tests for extract_metric_series function."""
    
    def test_extract_mean_metric(self):
        """Test extracting metric per episode (not aggregated by round)."""
        metrics_by_round = {
            0: [
                {"episode": 0, "p1_win_rate": 0.5},
                {"episode": 1, "p1_win_rate": 0.6},
                {"episode": 2, "p1_win_rate": 0.7}
            ],
            1: [
                {"episode": 0, "p1_win_rate": 0.6},
                {"episode": 1, "p1_win_rate": 0.7},
                {"episode": 2, "p1_win_rate": 0.8}
            ]
        }
        
        episodes, values, std_devs = extract_metric_series(metrics_by_round, "p1_win_rate", "mean")
        
        # Should return all episodes in order with cumulative episode numbers
        assert len(episodes) == 6
        assert len(values) == 6
        assert episodes == [0, 1, 2, 3, 4, 5]  # Cumulative episode numbers
        assert values == [0.5, 0.6, 0.7, 0.6, 0.7, 0.8]  # All episode values
        assert std_devs is None  # No std_devs for episode-based plotting
    
    def test_extract_final_metric(self):
        """Test extracting metric per episode (aggregation parameter ignored)."""
        metrics_by_round = {
            0: [
                {"episode": 0, "p1_win_rate": 0.5},
                {"episode": 1, "p1_win_rate": 0.6},
                {"episode": 2, "p1_win_rate": 0.7}
            ],
            1: [
                {"episode": 0, "p1_win_rate": 0.6},
                {"episode": 1, "p1_win_rate": 0.8}
            ]
        }
        
        episodes, values, std_devs = extract_metric_series(metrics_by_round, "p1_win_rate", "final")
        
        # Should return all episodes, not just final ones
        assert len(episodes) == 5
        assert len(values) == 5
        assert values == [0.5, 0.6, 0.7, 0.6, 0.8]  # All episode values
        assert std_devs is None
    
    def test_extract_missing_metric(self):
        """Test extracting metric that doesn't exist."""
        metrics_by_round = {
            0: [{"episode": 0, "p1_win_rate": 0.5}],
            1: [{"episode": 0, "p1_win_rate": 0.6}]
        }
        
        episodes, values, std_devs = extract_metric_series(metrics_by_round, "nonexistent_metric", "mean")
        
        # Should return empty lists if metric doesn't exist
        assert len(episodes) == 0
        assert len(values) == 0
    
    def test_extract_metric_with_none_values(self):
        """Test extracting metric with None values (filtered out)."""
        metrics_by_round = {
            0: [
                {"episode": 0, "loss": 0.1},
                {"episode": 1, "loss": None},
                {"episode": 2, "loss": 0.2}
            ]
        }
        
        episodes, values, std_devs = extract_metric_series(metrics_by_round, "loss", "mean")
        
        # Should only return episodes with valid values
        assert len(episodes) == 2
        assert len(values) == 2
        assert values == [0.1, 0.2]  # None filtered out


class TestCalculateStatistics(unittest.TestCase if not HAS_PYTEST else object):
    """Tests for calculate_statistics function."""
    
    def test_calculate_basic_statistics(self):
        """Test calculating basic statistics."""
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        stats = calculate_statistics(values)
        
        assert abs(stats["mean"] - 0.7) < 0.01
        assert stats["min"] == 0.5
        assert stats["max"] == 0.9
        assert abs(stats["median"] - 0.7) < 0.01
        assert stats["std"] > 0
    
    def test_calculate_statistics_with_trend(self):
        """Test calculating statistics including trend."""
        values = [0.5, 0.6, 0.7, 0.8, 0.9]  # Increasing trend
        stats = calculate_statistics(values)
        
        assert stats["trend"] > 0  # Positive trend
        assert 0 <= stats["r_squared"] <= 1
    
    def test_calculate_statistics_empty(self):
        """Test calculating statistics for empty list."""
        values = []
        stats = calculate_statistics(values)
        
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])
        assert np.isnan(stats["min"])
        assert np.isnan(stats["max"])
    
    def test_calculate_statistics_single_value(self):
        """Test calculating statistics for single value."""
        values = [0.5]
        stats = calculate_statistics(values)
        
        assert stats["mean"] == 0.5
        assert stats["min"] == 0.5
        assert stats["max"] == 0.5
        assert stats["trend"] == 0.0
        assert stats["r_squared"] == 0.0
    
    def test_calculate_statistics_with_nan(self):
        """Test calculating statistics with NaN values."""
        values = [0.5, np.nan, 0.7, np.nan, 0.9]
        stats = calculate_statistics(values)
        
        assert abs(stats["mean"] - 0.7) < 0.01  # Mean of [0.5, 0.7, 0.9]
        assert stats["min"] == 0.5
        assert stats["max"] == 0.9


class TestGenerateGraph(unittest.TestCase if not HAS_PYTEST else object):
    """Tests for generate_graph function."""
    
    @patch('generate_metrics_graphs.plt.savefig')
    @patch('generate_metrics_graphs.plt.close')
    @patch('generate_metrics_graphs.plt.subplots')
    @patch('generate_metrics_graphs.plt.tight_layout')
    @patch('generate_metrics_graphs.sns.set_style')
    def test_generate_basic_graph(self, mock_sns, mock_tight, mock_subplots, mock_close, mock_savefig):
        """Test generating a basic graph."""
        # Mock subplots to return a figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        # Mock figure.savefig method
        mock_fig.savefig = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episodes = [0, 1, 2, 3]
            values = [0.5, 0.6, 0.7, 0.8]
            
            result = generate_graph(
                episodes, values, "p1_win_rate", "selfplay", "hand_only",
                output_dir, "png", "seaborn"
            )
            
            assert result is not None
            # Since we're mocking savefig, the file won't actually exist, but the path should be correct
            assert result.parent.exists()  # Directory should exist
            assert result.name == "p1_win_rate_selfplay_hand_only.png"
            # The function uses fig.savefig(), not plt.savefig()
            mock_fig.savefig.assert_called_once()
            mock_close.assert_called_once()
    
    @patch('generate_metrics_graphs.plt.savefig')
    @patch('generate_metrics_graphs.plt.close')
    @patch('generate_metrics_graphs.plt.subplots')
    @patch('generate_metrics_graphs.plt.tight_layout')
    @patch('generate_metrics_graphs.sns.set_style')
    def test_generate_graph_with_error_bars(self, mock_sns, mock_tight, mock_subplots, mock_close, mock_savefig):
        """Test generating graph with error bars."""
        # Mock subplots to return a figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_fig.savefig = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episodes = [0, 1, 2]
            values = [0.5, 0.6, 0.7]
            std_devs = [0.05, 0.04, 0.03]  # Ignored for episode-based plotting
            
            result = generate_graph(
                episodes, values, "p1_win_rate", "selfplay", "hand_only",
                output_dir, "png", "seaborn", std_devs=std_devs
            )
            
            assert result is not None
            # The function uses fig.savefig(), not plt.savefig()
            mock_fig.savefig.assert_called_once()
    
    @patch('generate_metrics_graphs.plt.close')
    def test_generate_graph_with_all_nan(self, mock_close):
        """Test generating graph with all NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episodes = [0, 1, 2]
            values = [np.nan, np.nan, np.nan]
            
            result = generate_graph(
                episodes, values, "p1_win_rate", "selfplay", "hand_only",
                output_dir, "png", "seaborn"
            )
            
            assert result is None
            mock_close.assert_called_once()
    
    @patch('generate_metrics_graphs.plt.savefig')
    @patch('generate_metrics_graphs.plt.close')
    @patch('generate_metrics_graphs.plt.subplots')
    @patch('generate_metrics_graphs.plt.tight_layout')
    @patch('generate_metrics_graphs.sns.set_style')
    def test_generate_graph_with_statistics(self, mock_sns, mock_tight, mock_subplots, mock_close, mock_savefig):
        """Test generating graph with statistics."""
        # Mock subplots to return a figure and axes
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        mock_fig.savefig = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            episodes = [0, 1, 2, 3, 4]
            values = [0.5, 0.6, 0.7, 0.8, 0.9]
            statistics = {
                "mean": 0.7,
                "std": 0.158,
                "trend": 0.1,
                "r_squared": 0.99
            }
            
            result = generate_graph(
                episodes, values, "p1_win_rate", "selfplay", "hand_only",
                output_dir, "png", "seaborn", statistics=statistics
            )
            
            assert result is not None
            # The function uses fig.savefig(), not plt.savefig()
            mock_fig.savefig.assert_called_once()


class TestIntegration(unittest.TestCase if not HAS_PYTEST else object):
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_selfplay(self):
        """Test full pipeline from file to graph generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            log_dir = base_dir / "action_logs" / "hand_only"
            log_dir.mkdir(parents=True)
            output_dir = base_dir / "output"
            
            # Create test metrics file
            file_path = log_dir / "metrics_hand_only_round_0_selfplay.jsonl"
            with open(file_path, 'w') as f:
                for episode in range(5):
                    data = {
                        "episode": episode,
                        "p1_win_rate": 0.5 + episode * 0.05,
                        "loss": 0.1 - episode * 0.01
                    }
                    f.write(json.dumps(data) + "\n")
            
            # Aggregate metrics
            metrics_by_round = aggregate_phase_metrics("hand_only", "selfplay", base_dir)
            assert len(metrics_by_round) == 1
            assert 0 in metrics_by_round
            
            # Extract metric series (now returns episodes, not rounds)
            episodes, values, std_devs = extract_metric_series(
                metrics_by_round, "p1_win_rate", "mean"
            )
            assert len(episodes) == 5  # 5 episodes from round 0
            assert len(values) == 5
            assert all(v >= 0.5 for v in values)  # First value is 0.5, rest are > 0.5
            assert values[0] == 0.5  # First episode
            assert values[-1] > 0.5  # Last episode should be higher
            
            # Calculate statistics
            stats = calculate_statistics(values)
            assert not np.isnan(stats["mean"])
            
            # Generate graph (with mocking to avoid actual file I/O)
            with patch('generate_metrics_graphs.plt.savefig'), \
                 patch('generate_metrics_graphs.plt.close'), \
                 patch('generate_metrics_graphs.plt.subplots') as mock_subplots, \
                 patch('generate_metrics_graphs.plt.tight_layout'), \
                 patch('generate_metrics_graphs.sns.set_style'):
                # Mock subplots
                mock_fig = MagicMock()
                mock_ax = MagicMock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                
                result = generate_graph(
                    episodes, values, "p1_win_rate", "selfplay", "hand_only",
                    output_dir, "png", "seaborn", std_devs=std_devs, statistics=stats
                )
                assert result is not None


if __name__ == "__main__":
    if HAS_PYTEST:
        exit_code = pytest.main([__file__, "-v"])
        sys.exit(exit_code)
    else:
        # Run tests manually if pytest is not available
        # Convert pytest-style tests to unittest
        import unittest
        
        # Create test suite from all test classes
        test_classes = [
            TestParseMetricsFile,
            TestExtractRoundNumber,
            TestAggregatePhaseMetrics,
            TestExtractMetricSeries,
            TestCalculateStatistics,
            TestGenerateGraph,
            TestIntegration
        ]
        
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

