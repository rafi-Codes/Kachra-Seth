"""Synthetic data generation utilities for Blender-based rendering."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import subprocess
import sys


class BlenderScriptGenerator:
    """Generate Python scripts for Blender rendering."""
    
    @staticmethod
    def generate_stack_render_script(
        output_path: str,
        count: int,
        category: str = "banknotes",
        image_size: int = 384,
        stack_angle: str = "top_down",
        lighting: str = "natural",
        output_format: str = "PNG"
    ) -> str:
        """Generate a Blender Python script for rendering stack images.
        
        Args:
            output_path: Path to save the rendered image
            count: Number of items in the stack
            category: Category of objects
            image_size: Output image size
            stack_angle: Viewing angle
            lighting: Lighting condition
            output_format: Image format (PNG, JPEG)
            
        Returns:
            Blender Python script as string
        """
        script = f'''
import bpy
import random
import math
from pathlib import Path

# Clear existing scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Set up render settings
scene = bpy.context.scene
scene.render.resolution_x = {image_size}
scene.render.resolution_y = {image_size}
scene.render.image_settings.file_format = '{output_format}'
scene.render.filepath = '{output_path}'

# Create camera
bpy.ops.object.camera_add(location=(0, -5, 3))
camera = bpy.context.active_object
camera.rotation_euler = (math.radians(60), 0, 0)
scene.camera = camera

# Set up lighting
if '{lighting}' == 'natural':
    # Natural lighting with sun and ambient
    light_data = bpy.data.lights.new(name='Sun', type='SUN')
    light_data.energy = 3.0
    light_object = bpy.data.objects.new(name='Sun', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (2, -2, 5)
    light_object.rotation_euler = (math.radians(45), 0, math.radians(30))
elif '{lighting}' == 'bright':
    light_data = bpy.data.lights.new(name='Bright', type='SUN')
    light_data.energy = 5.0
    light_object = bpy.data.objects.new(name='Bright', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (0, -5, 10)
elif '{lighting}' == 'dim':
    light_data = bpy.data.lights.new(name='Dim', type='POINT')
    light_data.energy = 1.0
    light_object = bpy.data.objects.new(name='Dim', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (0, -3, 2)
elif '{lighting}' == 'backlit':
    light_data = bpy.data.lights.new(name='Back', type='POINT')
    light_data.energy = 4.0
    light_object = bpy.data.objects.new(name='Back', object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = (0, 2, 1)

# Create stack
# This is a simplified placeholder - actual implementation would depend on category
# For now, we create a simple cube stack as an example
for i in range({count}):
    bpy.ops.mesh.primitive_cube_add(
        size=0.1,
        location=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), i * 0.1)
    )
    obj = bpy.context.active_object
    # Add random rotation for realism
    obj.rotation_euler = (
        random.uniform(-0.1, 0.1),
        random.uniform(-0.1, 0.1),
        random.uniform(-0.5, 0.5)
    )

# Set camera angle based on stack_angle
if '{stack_angle}' == 'top_down':
    camera.location = (0, 0, 8)
    camera.rotation_euler = (0, 0, 0)
elif '{stack_angle}' == '45_degree':
    camera.location = (0, -6, 4)
    camera.rotation_euler = (math.radians(45), 0, 0)
elif '{stack_angle}' == 'side':
    camera.location = (6, 0, 2)
    camera.rotation_euler = (math.radians(90), 0, math.radians(90))

# Add ground plane
bpy.ops.mesh.primitive_plane_add(size=10)
ground = bpy.context.active_object
ground.location = (0, 0, -0.05)

# Render
bpy.ops.render.render(write_still=True)
print(f"Rendered to: {output_path}")
'''
        return script
    
    @staticmethod
    def generate_batch_script(
        output_dir: str,
        counts: List[int],
        categories: List[str],
        angles: List[str],
        lighting_conditions: List[str],
        images_per_config: int = 5
    ) -> str:
        """Generate a batch rendering script.
        
        Args:
            output_dir: Directory to save rendered images
            counts: List of count values to generate
            categories: List of categories
            angles: List of stack angles
            lighting_conditions: List of lighting conditions
            images_per_config: Number of images per configuration
            
        Returns:
            Blender Python script as string
        """
        script = f'''
import bpy
import random
import math
from pathlib import Path
import json

output_dir = Path('{output_dir}')
output_dir.mkdir(parents=True, exist_ok=True)

counts = {counts}
categories = {categories}
angles = {angles}
lighting_conditions = {lighting_conditions}
images_per_config = {images_per_config}

generated_data = []

# Rendering loop
for category in categories:
    for count in counts:
        for angle in angles:
            for lighting in lighting_conditions:
                for i in range(images_per_config):
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"synthetic_{{category}}_{{count}}_{{angle}}_{{lighting}}_{{timestamp}}.png"
                    output_path = output_dir / filename
                    
                    # Clear existing scene
                    bpy.ops.object.select_all(action='SELECT')
                    bpy.ops.object.delete()
                    
                    # Set up render settings
                    scene = bpy.context.scene
                    scene.render.resolution_x = 384
                    scene.render.resolution_y = 384
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.filepath = str(output_path)
                    
                    # Create camera and set angle
                    bpy.ops.object.camera_add(location=(0, -5, 3))
                    camera = bpy.context.active_object
                    
                    # Set camera angle
                    if angle == 'top_down':
                        camera.location = (0, 0, 8)
                        camera.rotation_euler = (0, 0, 0)
                    elif angle == '45_degree':
                        camera.location = (0, -6, 4)
                        camera.rotation_euler = (math.radians(45), 0, 0)
                    elif angle == 'side':
                        camera.location = (6, 0, 2)
                        camera.rotation_euler = (math.radians(90), 0, math.radians(90))
                    
                    scene.camera = camera
                    
                    # Set up lighting
                    if lighting == 'natural':
                        light_data = bpy.data.lights.new(name='Sun', type='SUN')
                        light_data.energy = 3.0
                        light_object = bpy.data.objects.new(name='Sun', object_data=light_data)
                        bpy.context.collection.objects.link(light_object)
                        light_object.location = (2, -2, 5)
                    elif lighting == 'bright':
                        light_data = bpy.data.lights.new(name='Bright', type='SUN')
                        light_data.energy = 5.0
                        light_object = bpy.data.objects.new(name='Bright', object_data=light_data)
                        bpy.context.collection.objects.link(light_object)
                        light_object.location = (0, -5, 10)
                    elif lighting == 'dim':
                        light_data = bpy.data.lights.new(name='Dim', type='POINT')
                        light_data.energy = 1.0
                        light_object = bpy.data.objects.new(name='Dim', object_data=light_data)
                        bpy.context.collection.objects.link(light_object)
                        light_object.location = (0, -3, 2)
                    elif lighting == 'backlit':
                        light_data = bpy.data.lights.new(name='Back', type='POINT')
                        light_data.energy = 4.0
                        light_object = bpy.data.objects.new(name='Back', object_data=light_data)
                        bpy.context.collection.objects.link(light_object)
                        light_object.location = (0, 2, 1)
                    
                    # Create stack (simplified - would need category-specific meshes)
                    for j in range(count):
                        bpy.ops.mesh.primitive_cube_add(
                            size=0.1,
                            location=(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), j * 0.1)
                        )
                        obj = bpy.context.active_object
                        obj.rotation_euler = (
                            random.uniform(-0.1, 0.1),
                            random.uniform(-0.1, 0.1),
                            random.uniform(-0.5, 0.5)
                        )
                    
                    # Add ground plane
                    bpy.ops.mesh.primitive_plane_add(size=10)
                    ground = bpy.context.active_object
                    ground.location = (0, 0, -0.05)
                    
                    # Render
                    bpy.ops.render.render(write_still=True)
                    
                    # Record metadata
                    generated_data.append({{
                        'image_id': filename,
                        'category': category,
                        'true_count': count,
                        'annotator_1_count': count,
                        'annotator_2_count': count,
                        'agreement_score': 1.0,
                        'stack_angle': angle,
                        'lighting': lighting,
                        'occlusion_percent': random.randint(0, 10),
                        'is_synthetic': True
                    }})
                    
                    print(f"Rendered {{filename}}")

# Save annotations
annotations_path = output_dir / 'synthetic_annotations.json'
with open(annotations_path, 'w') as f:
    json.dump(generated_data, f, indent=2)

print(f"Generated {{len(generated_data)}} synthetic images")
print(f"Annotations saved to {{annotations_path}}")
'''
        return script


class SyntheticDataGenerator:
    """Manage synthetic data generation process."""
    
    def __init__(self, blender_path: Optional[str] = None):
        """Initialize synthetic data generator.
        
        Args:
            blender_path: Path to Blender executable (optional)
        """
        self.blender_path = blender_path or self._find_blender()
    
    def _find_blender(self) -> Optional[str]:
        """Find Blender executable in system."""
        # Common Blender installation paths
        common_paths = [
            "blender",  # If in PATH
            "/Applications/Blender.app/Contents/MacOS/Blender",  # macOS
            "C:\\Program Files\\Blender Foundation\\Blender\\blender.exe",  # Windows
            "/usr/bin/blender",  # Linux
        ]
        
        for path in common_paths:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return None
    
    def generate_single_image(
        self,
        output_path: str,
        count: int,
        category: str = "banknotes",
        **kwargs
    ) -> bool:
        """Generate a single synthetic image.
        
        Args:
            output_path: Path to save the image
            count: Number of items
            category: Object category
            **kwargs: Additional parameters for rendering
            
        Returns:
            True if successful, False otherwise
        """
        if self.blender_path is None:
            raise RuntimeError("Blender not found. Please provide blender_path.")
        
        # Generate script
        script = BlenderScriptGenerator.generate_stack_render_script(
            output_path=output_path,
            count=count,
            category=category,
            **kwargs
        )
        
        # Save script to temp file
        script_path = Path(output_path).parent / "temp_render_script.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Execute Blender with script
        try:
            subprocess.run(
                [self.blender_path, "--background", "--python", str(script_path)],
                check=True,
                timeout=300
            )
            
            # Clean up script file
            script_path.unlink()
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Blender execution failed: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("Blender execution timed out")
            return False
    
    def generate_batch(
        self,
        output_dir: str,
        count_range: Tuple[int, int] = (200, 500),
        categories: Optional[List[str]] = None,
        images_per_config: int = 5
    ) -> List[Dict]:
        """Generate a batch of synthetic images.
        
        Args:
            output_dir: Directory to save images
            count_range: Range of counts to generate (min, max)
            categories: List of categories (None = all)
            images_per_config: Images per configuration
            
        Returns:
            List of generated annotation dictionaries
        """
        if self.blender_path is None:
            raise RuntimeError("Blender not found. Please provide blender_path.")
        
        categories = categories or ["banknotes", "books", "papers", "tiles", "cards", "plates"]
        counts = list(range(count_range[0], count_range[1] + 1, 50))  # Every 50 items
        angles = ["top_down", "45_degree", "side"]
        lighting_conditions = ["natural", "bright", "dim", "backlit"]
        
        # Generate batch script
        script = BlenderScriptGenerator.generate_batch_script(
            output_dir=output_dir,
            counts=counts,
            categories=categories,
            angles=angles,
            lighting_conditions=lighting_conditions,
            images_per_config=images_per_config
        )
        
        # Save script
        script_path = Path(output_dir) / "batch_render_script.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(script)
        
        # Execute Blender
        try:
            subprocess.run(
                [self.blender_path, "--background", "--python", str(script_path)],
                check=True,
                timeout=3600  # 1 hour timeout for batch
            )
            
            # Load generated annotations
            annotations_path = Path(output_dir) / "synthetic_annotations.json"
            if annotations_path.exists():
                with open(annotations_path, 'r') as f:
                    annotations = json.load(f)
                return annotations
            
            return []
            
        except subprocess.CalledProcessError as e:
            print(f"Blender batch execution failed: {e}")
            return []
        except subprocess.TimeoutExpired:
            print("Blender batch execution timed out")
            return []


class SyntheticAnnotationManager:
    """Manage annotations for synthetic data."""
    
    @staticmethod
    def merge_annotations(
        real_annotations: List[Dict],
        synthetic_annotations: List[Dict],
        output_path: str
    ):
        """Merge real and synthetic annotations.
        
        Args:
            real_annotations: Real image annotations
            synthetic_annotations: Synthetic image annotations
            output_path: Path to save merged annotations
        """
        # Add flags to distinguish synthetic data
        for ann in synthetic_annotations:
            ann['is_synthetic'] = True
        
        for ann in real_annotations:
            ann['is_synthetic'] = False
        
        merged = real_annotations + synthetic_annotations
        
        with open(output_path, 'w') as f:
            json.dump(merged, f, indent=2)
        
        print(f"Merged {len(real_annotations)} real and {len(synthetic_annotations)} synthetic annotations")
        print(f"Total: {len(merged)} annotations")
    
    @staticmethod
    def filter_synthetic(annotations: List[Dict]) -> List[Dict]:
        """Filter to get only synthetic annotations."""
        return [ann for ann in annotations if ann.get('is_synthetic', False)]
    
    @staticmethod
    def filter_real(annotations: List[Dict]) -> List[Dict]:
        """Filter to get only real annotations."""
        return [ann for ann in annotations if not ann.get('is_synthetic', False)]


def generate_synthetic_dataset(
    output_dir: str,
    count_range: Tuple[int, int] = (200, 500),
    blender_path: Optional[str] = None
) -> List[Dict]:
    """Convenience function to generate synthetic dataset.
    
    Args:
        output_dir: Directory to save synthetic data
        count_range: Range of counts to generate
        blender_path: Optional Blender executable path
        
    Returns:
        List of generated annotations
    """
    generator = SyntheticDataGenerator(blender_path)
    return generator.generate_batch(
        output_dir=output_dir,
        count_range=count_range
    )
