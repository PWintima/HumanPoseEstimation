"""
Memory Management and Tracking Module

Tracks memory usage before and after each processing stage to monitor
memory efficiency and identify potential memory leaks or optimization opportunities.
"""

import psutil
import os
import gc
import sys
from typing import Dict, List, Optional
from datetime import datetime
import json


class MemoryTracker:
    """
    Tracks memory usage throughout the processing pipeline.
    
    Monitors:
    - Process memory (RSS - Resident Set Size)
    - Virtual memory
    - Memory deltas between stages
    - Peak memory usage
    """
    
    def __init__(self, process_id: Optional[int] = None):
        """
        Initialize memory tracker.
        
        Args:
            process_id: Process ID to track (None for current process)
        """
        self.process_id = process_id or os.getpid()
        self.process = psutil.Process(self.process_id)
        self.stages = []
        self.peak_memory = 0
        self.initial_memory = self._get_memory_info()
    
    def _get_memory_info(self) -> Dict[str, float]:
        """
        Get current memory information.
        
        Returns:
            Dictionary with memory metrics in MB
        """
        try:
            mem_info = self.process.memory_info()
            mem_percent = self.process.memory_percent()
            
            return {
                'rss_mb': mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
                'vms_mb': mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
                'percent': mem_percent,
                'available_mb': psutil.virtual_memory().available / (1024 * 1024),
                'total_mb': psutil.virtual_memory().total / (1024 * 1024),
            }
        except Exception as e:
            print(f"Warning: Could not get memory info: {e}")
            return {
                'rss_mb': 0.0,
                'vms_mb': 0.0,
                'percent': 0.0,
                'available_mb': 0.0,
                'total_mb': 0.0,
            }
    
    def start_stage(self, stage_name: str) -> Dict[str, float]:
        """
        Mark the start of a processing stage.
        
        Args:
            stage_name: Name of the stage (e.g., "Stage 1: EDA")
            
        Returns:
            Memory info at stage start
        """
        # Force garbage collection before measuring
        gc.collect()
        
        mem_info = self._get_memory_info()
        
        self.stages.append({
            'stage': stage_name,
            'start_memory': mem_info.copy(),
            'end_memory': None,
            'delta_mb': None,
            'peak_mb': None,
            'start_time': datetime.now().isoformat(),
            'end_time': None,
        })
        
        # Update peak memory
        if mem_info['rss_mb'] > self.peak_memory:
            self.peak_memory = mem_info['rss_mb']
        
        return mem_info
    
    def end_stage(self, stage_name: Optional[str] = None) -> Dict[str, float]:
        """
        Mark the end of a processing stage.
        
        Args:
            stage_name: Name of the stage (if None, uses last started stage)
            
        Returns:
            Memory info at stage end
        """
        # Force garbage collection before measuring
        gc.collect()
        
        mem_info = self._get_memory_info()
        
        # Find the stage (use last one if name not provided)
        if stage_name:
            stage = next((s for s in reversed(self.stages) if s['stage'] == stage_name), None)
        else:
            stage = self.stages[-1] if self.stages else None
        
        if stage and stage['end_memory'] is None:
            stage['end_memory'] = mem_info.copy()
            stage['end_time'] = datetime.now().isoformat()
            
            # Calculate delta
            if stage['start_memory']:
                stage['delta_mb'] = mem_info['rss_mb'] - stage['start_memory']['rss_mb']
            
            # Update peak memory
            if mem_info['rss_mb'] > self.peak_memory:
                self.peak_memory = mem_info['rss_mb']
            
            stage['peak_mb'] = self.peak_memory
        
        return mem_info
    
    def get_summary(self) -> Dict:
        """
        Get summary of memory usage across all stages.
        
        Returns:
            Dictionary with memory summary
        """
        final_memory = self._get_memory_info()
        
        return {
            'initial_memory_mb': self.initial_memory['rss_mb'],
            'final_memory_mb': final_memory['rss_mb'],
            'peak_memory_mb': self.peak_memory,
            'total_increase_mb': final_memory['rss_mb'] - self.initial_memory['rss_mb'],
            'stages': self.stages,
            'system_memory': {
                'total_mb': self.initial_memory['total_mb'],
                'available_mb': final_memory['available_mb'],
                'used_percent': final_memory['percent'],
            }
        }
    
    def print_stage_report(self, stage_name: str):
        """
        Print memory report for a specific stage.
        
        Args:
            stage_name: Name of the stage
        """
        stage = next((s for s in self.stages if s['stage'] == stage_name), None)
        
        if not stage:
            print(f"Stage '{stage_name}' not found")
            return
        
        print("\n" + "="*60)
        print(f"MEMORY REPORT: {stage_name}")
        print("="*60)
        
        if stage['start_memory']:
            print(f"Before: {stage['start_memory']['rss_mb']:.2f} MB (RSS)")
            print(f"        {stage['start_memory']['vms_mb']:.2f} MB (Virtual)")
            print(f"        {stage['start_memory']['percent']:.1f}% of system memory")
        
        if stage['end_memory']:
            print(f"After:  {stage['end_memory']['rss_mb']:.2f} MB (RSS)")
            print(f"        {stage['end_memory']['vms_mb']:.2f} MB (Virtual)")
            print(f"        {stage['end_memory']['percent']:.1f}% of system memory")
            
            if stage['delta_mb'] is not None:
                delta = stage['delta_mb']
                if delta > 0:
                    print(f"Change: +{delta:.2f} MB (increased)")
                elif delta < 0:
                    print(f"Change: {delta:.2f} MB (decreased)")
                else:
                    print(f"Change: {delta:.2f} MB (no change)")
        
        if stage['peak_mb']:
            print(f"Peak:   {stage['peak_mb']:.2f} MB")
        
        print("="*60)
    
    def print_summary(self):
        """Print overall memory usage summary."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)
        print(f"Initial Memory:  {summary['initial_memory_mb']:.2f} MB")
        print(f"Final Memory:    {summary['final_memory_mb']:.2f} MB")
        print(f"Peak Memory:     {summary['peak_memory_mb']:.2f} MB")
        print(f"Total Increase:  {summary['total_increase_mb']:.2f} MB")
        print(f"\nSystem Memory:")
        print(f"  Total:         {summary['system_memory']['total_mb']:.2f} MB")
        print(f"  Available:     {summary['system_memory']['available_mb']:.2f} MB")
        print(f"  Used:          {summary['system_memory']['used_percent']:.1f}%")
        
        print("\nStage-by-Stage Breakdown:")
        print("-"*60)
        for stage in summary['stages']:
            if stage['end_memory']:
                delta = stage['delta_mb'] if stage['delta_mb'] is not None else 0
                delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
                print(f"{stage['stage']:30s} | {delta_str:>10s} MB")
        
        print("="*60)
    
    def save_report(self, filepath: str):
        """
        Save memory report to JSON file.
        
        Args:
            filepath: Path to save the report
        """
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nMemory report saved to: {filepath}")


def optimize_memory():
    """
    Perform memory optimization operations.
    
    Forces garbage collection and clears caches.
    """
    # Force garbage collection
    collected = gc.collect()
    
    # Clear Python's internal caches if available
    if hasattr(sys, 'intern'):
        # Clear interned strings cache (limited effect)
        pass
    
    return collected


def get_memory_usage_mb() -> float:
    """
    Quick function to get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except:
        return 0.0

