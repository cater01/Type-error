#!/usr/bin/env python3
"""
分别计算每个 .jsonl 文件中 elapsed_seconds 字段的平均值
用法: python calc_avg.py file1.jsonl file2.jsonl file3.jsonl ...
"""

import json
import sys
import os


# Shared CLI/config support; works when invoked from any working directory.
import sys as _sys
from pathlib import Path as _Path
from pathlib import Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from runtime import (REPO_ROOT, ConfigArgumentParser, generation_datasets,
                     external_root, configure_java, finish_setup, seed_torch,
                     credential, repo_path)

def calculate_single_file_average(file_path):
    """
    计算单个文件中 elapsed_seconds 的平均值
    
    Returns:
        dict: 包含文件名、平均值、有效记录数等信息
    """
    total_time = 0.0
    valid_count = 0
    missing_count = 0
    errors = []
    
    if not os.path.exists(file_path):
        return {
            'file': file_path,
            'average': None,
            'valid_count': 0,
            'missing_count': 0,
            'error': f"文件不存在: {file_path}"
        }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    if 'elapsed_seconds' in data:
                        try:
                            elapsed = float(data['elapsed_seconds'])
                            total_time += elapsed
                            valid_count += 1
                        except (ValueError, TypeError):
                            missing_count += 1
                    else:
                        missing_count += 1
                except json.JSONDecodeError:
                    missing_count += 1
                    
    except Exception as e:
        return {
            'file': file_path,
            'average': None,
            'valid_count': 0,
            'missing_count': 0,
            'error': f"读取文件失败: {e}"
        }
    
    average = total_time / valid_count if valid_count > 0 else None
    
    return {
        'file': file_path,
        'average': average,
        'valid_count': valid_count,
        'missing_count': missing_count,
        'total_time': total_time,
        'error': None
    }

def main():
    parser = ConfigArgumentParser(description='Calculate mean elapsed_seconds from generation JSONL files.')
    parser.add_argument('files', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, default=None, help='Optional text report output.')
    args = parser.parse_args()
    finish_setup(args, args.out.with_suffix('.run.json') if args.out else None)
    if args.out:
        import contextlib
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('w', encoding='utf-8') as handle, contextlib.redirect_stdout(handle):
            return calculate_times(args.files)
    return calculate_times(args.files)

def calculate_times(file_paths):
    
    if not file_paths:
        print("错误: 请至少指定一个 .jsonl 文件路径")
        print(f"用法: python {os.path.basename(__file__)} file1.jsonl file2.jsonl ...")
        sys.exit(1)
    
    print(f"共处理 {len(file_paths)} 个文件")
    print("=" * 70)
    
    all_results = []
    grand_total_time = 0
    grand_valid_count = 0
    
    # 分别处理每个文件
    for file_path in file_paths:
        result = calculate_single_file_average(file_path)
        all_results.append(result)
        
        # 显示单个文件结果
        print(f"\n📁 文件: {result['file']}")
        if result['error']:
            print(f"   ❌ {result['error']}")
        else:
            print(f"   ✅ 有效记录: {result['valid_count']}")
            print(f"   ⚠️  无效/缺失: {result['missing_count']}")
            if result['average'] is not None:
                print(f"   📈 平均耗时: {result['average']:.6f} 秒")
                print(f"   📊 总耗时: {result['total_time']:.4f} 秒")
                grand_total_time += result['total_time']
                grand_valid_count += result['valid_count']
            else:
                print(f"   ⚠️  无有效数据")
        print("-" * 70)
    
    # 计算总体平均值
    overall_average = grand_total_time / grand_valid_count if grand_valid_count > 0 else None
    
    # 显示汇总
    print("\n" + "=" * 70)
    print("📊 汇总统计")
    print(f"   ✅ 总计有效记录: {grand_valid_count}")
    if overall_average is not None:
        print(f"   📈 总体平均耗时: {overall_average:.6f} 秒")
        print(f"   📊 总耗时: {grand_total_time:.4f} 秒")
    else:
        print("   ⚠️  没有有效数据")
    
    # 显示每个文件的平均值对比
    print("\n📊 各文件平均值对比:")
    print("-" * 70)
    for result in all_results:
        if result['average'] is not None:
            print(f"   {os.path.basename(result['file'])}: {result['average']:.6f} 秒")
        else:
            print(f"   {os.path.basename(result['file'])}: 无数据")
    print("=" * 70)

if __name__ == "__main__":
    main()
