"""Data visualization and exploration tools."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from PIL import Image
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class DataVisualizer:
    """Visualize dataset statistics and samples."""
    
    @staticmethod
    def plot_count_distribution(
        annotations: List[Dict],
        save_path: Optional[str] = None,
        bins: int = 50
    ):
        """Plot distribution of item counts.
        
        Args:
            annotations: List of annotation dictionaries
            save_path: Optional path to save the plot
            bins: Number of bins for histogram
        """
        counts = [ann['true_count'] for ann in annotations]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(counts, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Count', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Distribution of Item Counts', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(counts, vert=True)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Count Distribution (Box Plot)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved count distribution plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_category_distribution(
        annotations: List[Dict],
        save_path: Optional[str] = None
    ):
        """Plot distribution of categories.
        
        Args:
            annotations: List of annotation dictionaries
            save_path: Optional path to save the plot
        """
        categories = [ann['category'] for ann in annotations]
        unique_cats, cat_counts = np.unique(categories, return_counts=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Bar chart
        bars = ax1.bar(unique_cats, cat_counts, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Category', fontsize=12)
        ax1.set_ylabel('Number of Images', fontsize=12)
        ax1.set_title('Distribution by Category', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        # Pie chart
        ax2.pie(cat_counts, labels=unique_cats, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Category Distribution', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved category distribution plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_agreement_score_distribution(
        annotations: List[Dict],
        save_path: Optional[str] = None
    ):
        """Plot distribution of annotator agreement scores.
        
        Args:
            annotations: List of annotation dictionaries
            save_path: Optional path to save the plot
        """
        agreements = [ann['agreement_score'] for ann in annotations]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(agreements, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Agreement Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Annotator Agreement Scores', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_agreement = np.mean(agreements)
        ax.axvline(mean_agreement, color='red', linestyle='--', 
                  label=f'Mean: {mean_agreement:.3f}')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved agreement score distribution plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_attribute_distributions(
        annotations: List[Dict],
        save_path: Optional[str] = None
    ):
        """Plot distributions of various attributes (lighting, angle, occlusion).
        
        Args:
            annotations: List of annotation dictionaries
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Lighting distribution
        lighting = [ann['lighting'] for ann in annotations]
        unique_light, light_counts = np.unique(lighting, return_counts=True)
        axes[0, 0].bar(unique_light, light_counts, color='lightgreen', edgecolor='black')
        axes[0, 0].set_title('Lighting Conditions')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Stack angle distribution
        angles = [ann['stack_angle'] for ann in annotations]
        unique_angles, angle_counts = np.unique(angles, return_counts=True)
        axes[0, 1].bar(unique_angles, angle_counts, color='lightblue', edgecolor='black')
        axes[0, 1].set_title('Stack Angles')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Occlusion distribution
        occlusions = [ann['occlusion_percent'] for ann in annotations]
        axes[1, 0].hist(occlusions, bins=20, color='orange', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Occlusion Percentage')
        axes[1, 0].set_xlabel('Occlusion %')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Count vs Category scatter
        categories = [ann['category'] for ann in annotations]
        counts = [ann['true_count'] for ann in annotations]
        unique_cats = list(set(categories))
        for cat in unique_cats:
            cat_counts = [count for cat_, count in zip(categories, counts) if cat_ == cat]
            axes[1, 1].scatter([cat] * len(cat_counts), cat_counts, alpha=0.5, label=cat)
        axes[1, 1].set_title('Count Distribution by Category')
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attribute distributions plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_sample_images(
        image_dir: str,
        annotations: List[Dict],
        num_samples: int = 9,
        save_path: Optional[str] = None
    ):
        """Plot sample images with their metadata.
        
        Args:
            image_dir: Directory containing images
            annotations: List of annotation dictionaries
            num_samples: Number of samples to display
            save_path: Optional path to save the plot
        """
        image_dir = Path(image_dir)
        samples = np.random.choice(annotations, min(num_samples, len(annotations)), replace=False)
        
        cols = 3
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, ann in enumerate(samples):
            row = idx // cols
            col = idx % cols
            
            # Load image
            image_path = image_dir / ann['image_id']
            if not image_path.exists():
                # Try alternative path
                image_path = image_dir / ann['category'] / ann['image_id']
            
            try:
                img = Image.open(image_path)
                axes[row, col].imshow(img)
            except:
                axes[row, col].text(0.5, 0.5, 'Image not found', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
            
            # Add metadata
            title = f"{ann['category']}\nCount: {ann['true_count']}\n{ann['lighting']}, {ann['stack_angle']}"
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].axis('off')
        
        # Remove empty subplots
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved sample images plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


class DatasetExplorer:
    """Explore and analyze dataset characteristics."""
    
    @staticmethod
    def generate_summary_report(
        annotations: List[Dict],
        output_path: Optional[str] = None
    ) -> Dict:
        """Generate a comprehensive summary report.
        
        Args:
            annotations: List of annotation dictionaries
            output_path: Optional path to save report
            
        Returns:
            Dictionary containing summary statistics
        """
        counts = [ann['true_count'] for ann in annotations]
        categories = [ann['category'] for ann in annotations]
        agreements = [ann['agreement_score'] for ann in annotations]
        
        report = {
            'total_images': len(annotations),
            'count_statistics': {
                'mean': float(np.mean(counts)),
                'std': float(np.std(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts)),
                'median': float(np.median(counts)),
                'percentiles': {
                    '25th': float(np.percentile(counts, 25)),
                    '50th': float(np.percentile(counts, 50)),
                    '75th': float(np.percentile(counts, 75)),
                    '90th': float(np.percentile(counts, 90)),
                    '95th': float(np.percentile(counts, 95))
                }
            },
            'category_distribution': dict(zip(*np.unique(categories, return_counts=True))),
            'agreement_statistics': {
                'mean': float(np.mean(agreements)),
                'std': float(np.std(agreements)),
                'min': float(np.min(agreements)),
                'max': float(np.max(agreements))
            },
            'attribute_distributions': {
                'lighting': dict(zip(*np.unique([ann['lighting'] for ann in annotations], return_counts=True))),
                'stack_angle': dict(zip(*np.unique([ann['stack_angle'] for ann in annotations], return_counts=True)))
            }
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Saved summary report to {output_path}")
        
        return report
    
    @staticmethod
    def print_summary_report(report: Dict):
        """Print a formatted summary report.
        
        Args:
            report: Summary report dictionary
        """
        print("=" * 60)
        print("DATASET SUMMARY REPORT")
        print("=" * 60)
        
        print(f"\nTotal Images: {report['total_images']}")
        
        print("\n--- Count Statistics ---")
        count_stats = report['count_statistics']
        print(f"Mean: {count_stats['mean']:.2f}")
        print(f"Std: {count_stats['std']:.2f}")
        print(f"Min: {count_stats['min']}")
        print(f"Max: {count_stats['max']}")
        print(f"Median: {count_stats['median']:.2f}")
        print(f"25th percentile: {count_stats['percentiles']['25th']:.2f}")
        print(f"75th percentile: {count_stats['percentiles']['75th']:.2f}")
        print(f"95th percentile: {count_stats['percentiles']['95th']:.2f}")
        
        print("\n--- Category Distribution ---")
        for cat, count in report['category_distribution'].items():
            print(f"{cat}: {count} images ({count/report['total_images']*100:.1f}%)")
        
        print("\n--- Annotator Agreement ---")
        agr_stats = report['agreement_statistics']
        print(f"Mean agreement: {agr_stats['mean']:.3f}")
        print(f"Std: {agr_stats['std']:.3f}")
        print(f"Range: [{agr_stats['min']:.3f}, {agr_stats['max']:.3f}]")
        
        print("\n--- Attribute Distributions ---")
        print("Lighting:")
        for lighting, count in report['attribute_distributions']['lighting'].items():
            print(f"  {lighting}: {count}")
        
        print("Stack Angles:")
        for angle, count in report['attribute_distributions']['stack_angle'].items():
            print(f"  {angle}: {count}")
        
        print("=" * 60)
    
    @staticmethod
    def identify_data_gaps(
        annotations: List[Dict],
        target_counts_per_category: int = 500,
        target_count_ranges: List[Tuple[int, int]] = [(5, 50), (51, 150), (151, 500)]
    ) -> Dict:
        """Identify gaps in dataset coverage.
        
        Args:
            annotations: List of annotation dictionaries
            target_counts_per_category: Target number of images per category
            target_count_ranges: Target count ranges to ensure coverage
            
        Returns:
            Dictionary identifying gaps
        """
        gaps = {}
        
        # Check category coverage
        category_counts = {}
        for ann in annotations:
            cat = ann['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        gaps['category_gaps'] = {
            cat: target_counts_per_category - count 
            for cat, count in category_counts.items()
            if count < target_counts_per_category
        }
        
        # Check count range coverage by category
        gaps['count_range_gaps'] = {}
        for cat in category_counts.keys():
            cat_annotations = [ann for ann in annotations if ann['category'] == cat]
            cat_counts = [ann['true_count'] for ann in cat_annotations]
            
            range_coverage = {}
            for min_c, max_c in target_count_ranges:
                in_range = sum(1 for c in cat_counts if min_c <= c <= max_c)
                range_coverage[f"{min_c}-{max_c}"] = in_range
            
            gaps['count_range_gaps'][cat] = range_coverage
        
        # Check attribute coverage
        lighting_coverage = {}
        for ann in annotations:
            lighting = ann['lighting']
            lighting_coverage[lighting] = lighting_coverage.get(lighting, 0) + 1
        
        angle_coverage = {}
        for ann in annotations:
            angle = ann['stack_angle']
            angle_coverage[angle] = angle_coverage.get(angle, 0) + 1
        
        gaps['attribute_gaps'] = {
            'lighting': lighting_coverage,
            'stack_angles': angle_coverage
        }
        
        return gaps


def visualize_dataset(
    annotations: List[Dict],
    image_dir: str,
    output_dir: str,
    generate_plots: bool = True
):
    """Generate all visualizations for a dataset.
    
    Args:
        annotations: List of annotation dictionaries
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        generate_plots: Whether to generate plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate summary report
    report = DatasetExplorer.generate_summary_report(
        annotations,
        output_path=str(output_dir / 'summary_report.json')
    )
    DatasetExplorer.print_summary_report(report)
    
    if generate_plots:
        # Generate all plots
        DataVisualizer.plot_count_distribution(
            annotations,
            save_path=str(output_dir / 'count_distribution.png')
        )
        
        DataVisualizer.plot_category_distribution(
            annotations,
            save_path=str(output_dir / 'category_distribution.png')
        )
        
        DataVisualizer.plot_agreement_score_distribution(
            annotations,
            save_path=str(output_dir / 'agreement_distribution.png')
        )
        
        DataVisualizer.plot_attribute_distributions(
            annotations,
            save_path=str(output_dir / 'attribute_distributions.png')
        )
        
        DataVisualizer.plot_sample_images(
            image_dir,
            annotations,
            num_samples=12,
            save_path=str(output_dir / 'sample_images.png')
        )
        
        print(f"\nAll visualizations saved to {output_dir}")
