"""
Metrics tracking for quantum circuit simulation.
"""
import time
import psutil
import os
from collections import defaultdict
from typing import Dict, Any, Optional


class CircuitMetrics:
    """
    Handles performance metrics tracking for quantum circuit simulation.
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize metrics tracker.
        
        Args:
            enabled: Whether to track metrics
        """
        self.enabled = enabled
        self._process = None
        self.metrics = None
        
        if self.enabled:
            self._initialize()
    
    def _initialize(self):
        """Initialize metrics tracking structures."""
        try:
            self._process = psutil.Process(os.getpid())
        except:
            self._process = None
        
        self.metrics = {
            # Execution timing
            'execution_start_time': None,
            'execution_end_time': None,
            'total_execution_time': 0.0,
            
            # Memory tracking (in MB)
            'initial_memory_mb': 0.0,
            'peak_memory_mb': 0.0,
            'final_memory_mb': 0.0,
            
            # Operation counts
            'gate_count': 0,
            'measurement_count': 0,
            'channel_count': 0,
            
            # Timing breakdowns
            'total_gate_time': 0.0,
            'total_measurement_time': 0.0,
            'total_channel_time': 0.0,
            
            # Error channel statistics
            'channels_applied': defaultdict(lambda: {
                'count': 0,
                'total_time': 0.0,
                'kraus_outcomes': []
            }),
            
            # Detailed operation log (optional)
            'operation_log': []
        }
    
    def start_execution(self):
        """Mark the start of circuit execution."""
        if not self.enabled:
            return
        
        self.metrics['execution_start_time'] = time.perf_counter()
        self.metrics['initial_memory_mb'] = self._get_memory_mb()
        self.metrics['peak_memory_mb'] = self.metrics['initial_memory_mb']
    
    def end_execution(self):
        """Mark the end of circuit execution."""
        if not self.enabled:
            return
        
        self.metrics['execution_end_time'] = time.perf_counter()
        self.metrics['final_memory_mb'] = self._get_memory_mb()
        self.metrics['total_execution_time'] = (
            self.metrics['execution_end_time'] - self.metrics['execution_start_time']
        )
    
    def record_gate(self, duration: float):
        """Record a gate operation."""
        if not self.enabled:
            return
        
        self.metrics['gate_count'] += 1
        self.metrics['total_gate_time'] += duration
        self.metrics['peak_memory_mb'] = max(
            self.metrics['peak_memory_mb'], 
            self._get_memory_mb()
        )
    
    def record_measurement(self, duration: float):
        """Record a measurement operation."""
        if not self.enabled:
            return
        
        self.metrics['measurement_count'] += 1
        self.metrics['total_measurement_time'] += duration
        self.metrics['peak_memory_mb'] = max(
            self.metrics['peak_memory_mb'], 
            self._get_memory_mb()
        )
    
    def record_channel(self, channel_name: str, duration: float = 0.0, 
                      kraus_index: Optional[int] = None):
        """
        Record an error channel application.
        
        Args:
            channel_name: Name of the error channel
            duration: Time taken to apply the channel (seconds)
            kraus_index: Which Kraus operator was selected
        """
        if not self.enabled:
            return
        
        stats = self.metrics['channels_applied'][channel_name]
        stats['count'] += 1
        stats['total_time'] += duration
        if kraus_index is not None:
            stats['kraus_outcomes'].append(kraus_index)
        
        self.metrics['channel_count'] += 1
        self.metrics['total_channel_time'] += duration
    
    def get_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of collected metrics.
        
        Returns:
            Dictionary containing metrics summary, or None if disabled
        """
        if not self.enabled or self.metrics is None:
            return None
        
        total_ops = (self.metrics['gate_count'] +
                    self.metrics['measurement_count'] +
                    self.metrics['channel_count'])
        
        summary = {
            'execution': {
                'total_time_seconds': self.metrics['total_execution_time'],
                'start_time': self.metrics['execution_start_time'],
                'end_time': self.metrics['execution_end_time'],
            },
            'memory': {
                'initial_mb': self.metrics['initial_memory_mb'],
                'peak_mb': self.metrics['peak_memory_mb'],
                'final_mb': self.metrics['final_memory_mb'],
                'delta_mb': self.metrics['final_memory_mb'] - self.metrics['initial_memory_mb'],
            },
            'operations': {
                'gate_count': self.metrics['gate_count'],
                'measurement_count': self.metrics['measurement_count'],
                'channel_count': self.metrics['channel_count'],
                'total_operations': total_ops,
            },
            'timing': {
                'gate_time_seconds': self.metrics['total_gate_time'],
                'measurement_time_seconds': self.metrics['total_measurement_time'],
                'channel_time_seconds': self.metrics['total_channel_time'],
                'gate_time_percent': self._safe_percent(
                    self.metrics['total_gate_time'],
                    self.metrics['total_execution_time']
                ),
                'measurement_time_percent': self._safe_percent(
                    self.metrics['total_measurement_time'],
                    self.metrics['total_execution_time']
                ),
                'channel_time_percent': self._safe_percent(
                    self.metrics['total_channel_time'],
                    self.metrics['total_execution_time']
                ),
            },
            'channels': {}
        }
        
        # Add channel details
        for name, stats in self.metrics['channels_applied'].items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
            summary['channels'][name] = {
                'hit_count': stats['count'],
                'total_time_seconds': stats['total_time'],
                'avg_time_seconds': avg_time,
                'kraus_outcomes': stats['kraus_outcomes'] if stats['kraus_outcomes'] else None,
            }
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        if self.enabled:
            self._initialize()
    
    def print_summary(self):
        """Print a formatted metrics summary to console."""
        if not self.enabled:
            print("Metrics tracking is disabled.")
            return
        
        summary = self.get_summary()
        if summary is None:
            print("No metrics available")
            return
        
        print("\n" + "="*70)
        print("QUANTUM SIMULATOR METRICS")
        print("="*70)
        
        print("\n[EXECUTION]")
        print(f"  Total execution time: {summary['execution']['total_time_seconds']:.6f} seconds")
        
        print("\n[MEMORY]")
        mem = summary['memory']
        print(f"  Initial: {mem['initial_mb']:.2f} MB")
        print(f"  Peak:    {mem['peak_mb']:.2f} MB")
        print(f"  Final:   {mem['final_mb']:.2f} MB")
        print(f"  Delta:   {mem['delta_mb']:+.2f} MB")
        
        print("\n[OPERATIONS]")
        ops = summary['operations']
        print(f"  Gates:        {ops['gate_count']}")
        print(f"  Measurements: {ops['measurement_count']}")
        print(f"  Channels:     {ops['channel_count']}")
        print(f"  Total:        {ops['total_operations']}")
        
        print("\n[TIMING BREAKDOWN]")
        timing = summary['timing']
        print(f"  Gates:        {timing['gate_time_seconds']:.6f}s ({timing['gate_time_percent']:.1f}%)")
        print(f"  Measurements: {timing['measurement_time_seconds']:.6f}s ({timing['measurement_time_percent']:.1f}%)")
        print(f"  Channels:     {timing['channel_time_seconds']:.6f}s ({timing['channel_time_percent']:.1f}%)")
        
        if summary['channels']:
            print("\n[ERROR CHANNELS]")
            for name, stats in summary['channels'].items():
                print(f"  {name}:")
                print(f"    Hit count:  {stats['hit_count']}")
                print(f"    Total time: {stats['total_time_seconds']:.6f}s")
                print(f"    Avg time:   {stats['avg_time_seconds']:.6f}s")
                if stats['kraus_outcomes']:
                    total = len(stats['kraus_outcomes'])
                    unique = set(stats['kraus_outcomes'])
                    dist = {k: stats['kraus_outcomes'].count(k)/total for k in unique}
                    print(f"    Kraus dist: {dist}")
        
        print("\n" + "="*70)
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if not self.enabled or self._process is None:
            return 0.0
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    @staticmethod
    def _safe_percent(part: float, total: float) -> float:
        """Calculate percentage safely (handles division by zero)."""
        if total == 0:
            return 0.0
        return (part / total) * 100.0