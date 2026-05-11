import os
import sys
import time
import yaml
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import itk
from DWInode_functions import import_images
from DWI_functions_standalone import generate_ADC_standalone

# Custom JSON encoder for compact array formatting
class CompactJSONEncoder(json.JSONEncoder):
    def encode(self, o):
        if isinstance(o, dict):
            items = []
            for k, v in o.items():
                k_json = json.dumps(k)
                if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                    v_json = '[' + ', '.join(json.dumps(x) for x in v) + ']'
                else:
                    v_json = self.encode(v)
                items.append(f'{k_json}: {v_json}')
            return '{' + ', '.join(items) + '}'
        elif isinstance(o, list) and all(isinstance(x, (int, float)) for x in o):
            return '[' + ', '.join(json.dumps(x) for x in o) + ']'
        return super().encode(o)
    
    def iterencode(self, o, _one_shot=False):
        for chunk in super().iterencode(o, _one_shot):
            yield chunk

# Configuration
DATA_LIST_PATH = 'recon_data.yaml'
# ROOT_OUTPUT_PATH = r'/mnt/rtstorage/Xiao/MotionCorrection/run_compare'
ROOT_OUTPUT_PATH = r'D:/run_compare'  # For local testing

# Parameter file paths
PARAM_STEP2_BSPLINE = Path("ElastixModelZoo/models/Par0029/Par0029-step2-bspline.txt")
PARAM_STEP2_INVERSION = Path("ElastixModelZoo/models/Par0029/Par0029-step2-bspline-inversion.txt")
PARAM_STEP3_BSPLINE = Path("ElastixModelZoo/models/Par0029/Par0029-step3-bspline.txt")

# Load configuration
with open(DATA_LIST_PATH, 'r') as f:
    config = yaml.safe_load(f)

def to_wsl_path(path):
    """Convert a Windows drive path to a WSL mount path when needed."""
    if os.name != 'nt' and len(path) >= 3 and path[1] == ':' and path[2] in ('/', '\\'):
        drive = path[0].lower()
        remainder = path[2:].replace('\\', '/')
        return f'/mnt/{drive}{remainder}'
    return path.replace('\\', '/')

def read_image_itk(array_data, pixel_type=itk.F):
    """Convert numpy array to ITK image."""
    img = itk.image_from_array(array_data.astype(np.float32))
    return img


def write_identity_transform(path: Path, image):
    """Write an identity transform parameter file matching the image geometry."""
    size = image.GetLargestPossibleRegion().GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = list(itk.GetArrayFromMatrix(image.GetDirection()).ravel())

    def fmt(seq):
        return " ".join(f"{v:.10f}" for v in seq)

    content = [
        '(Transform "TranslationTransform")',
        '(NumberOfParameters 3)',
        '(TransformParameters 0. 0. 0.)',
        '(InitialTransformParametersFileName "NoInitialTransform")',
        '(HowToCombineTransforms "Compose")',
        '(FixedImageDimension 3)',
        '(MovingImageDimension 3)',
        '(FixedInternalImagePixelType "float")',
        '(MovingInternalImagePixelType "float")',
        f"(Index 0 0 0)",
        f"(Size {size[0]} {size[1]} {size[2]})",
        f"(Spacing {fmt(spacing)})",
        f"(Origin {fmt(origin)})",
        f"(Direction {fmt(direction)})",
        '(UseDirectionCosines "true")',
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        '(FinalBSplineInterpolationOrder 1)',
        '(Resampler "DefaultResampler")',
        '(DefaultPixelValue -1.000000)',
        '(ResultImageFormat "nii")',
        '(ResultImagePixelType "float")',
        '(CompressResultImage "false")',
    ]

    path.write_text("\n".join(content))


def write_weighted_combination(path: Path, weights, subtransforms, image):
    """Write a weighted-combination transform parameter file for transformix."""
    if len(weights) != len(subtransforms):
        raise ValueError("weights and subtransforms length mismatch")

    size = image.GetLargestPossibleRegion().GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = list(itk.GetArrayFromMatrix(image.GetDirection()).ravel())

    def fmt(seq):
        return " ".join(f"{v:.10f}" for v in seq)

    weights_str = " ".join(f"{w:.6f}" for w in weights)
    sub_paths = " ".join(f'"{p}"' for p in subtransforms)

    content = [
        '(Transform "WeightedCombinationTransform")',
        f"(NumberOfParameters {len(weights)})",
        f"(TransformParameters {weights_str})",
        f"(SubTransforms {sub_paths})",
        '(AutomaticScalesEstimation "true")',
        '(FixedImageDimension 3)',
        '(MovingImageDimension 3)',
        '(FixedInternalImagePixelType "float")',
        '(MovingInternalImagePixelType "float")',
        f"(Index 0 0 0)",
        f"(Size {size[0]} {size[1]} {size[2]})",
        f"(Spacing {fmt(spacing)})",
        f"(Origin {fmt(origin)})",
        f"(Direction {fmt(direction)})",
        '(UseDirectionCosines "true")',
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        '(FinalBSplineInterpolationOrder 3)',
        '(Resampler "DefaultResampler")',
        '(DefaultPixelValue -1.000000)',
        '(ResultImageFormat "nii")',
        '(ResultImagePixelType "float")',
        '(CompressResultImage "false")',
    ]

    path.write_text("\n".join(content))


def write_parameter_map_to_file(param_map, path: Path):
    """Write a parameter map dictionary to a file in elastix format."""
    lines = []
    for key, values in param_map.items():
        if isinstance(values, (list, tuple)):
            # Join multiple values with spaces
            values_str = " ".join(str(v) for v in values)
        else:
            values_str = str(values)
        lines.append(f'({key} {values_str})')
    
    path.write_text("\n".join(lines))


def save_parameter_object(po: object, path: Path):
    """Save a ParameterObject to file using its output directory method."""
    # elastix writes TransformParameters.0.txt in the output directory
    # So we create a temp output dir, let it write there, then copy
    import shutil
    temp_output = Path(path.parent) / f"temp_po_{path.stem}"
    temp_output.mkdir(exist_ok=True)
    
    # This should write to temp_output with various output files
    # including TransformParameters.0.txt
    try:
        po.WriteParameterFile()  # No-arg version
    except:
        pass
    
    # Try to find and copy the transform file
    result_file = temp_output / "TransformParameters.0.txt"
    if result_file.exists():
        shutil.copy(result_file, path)
        shutil.rmtree(temp_output)
    else:
        # Fallback: elastix_registration_method already wrote the files to a temp location
        # Just write the parameter object as-is by reading its content
        raise FileNotFoundError(f"Expected {result_file} not found after WriteParameterFile()")


def register_to_midpoint(
    volumes,
    param_step2: Path,
    param_step2_inversion: Path,
    temp_dir: Path,
    output_dir: Path = None,
    labels=None,
):
    """
    Register volumes to a midpoint space (Par0029 Step 2, following Step 1 workflow).

    For each volume i:
    1. Register all volumes j (including i itself via identity) to i using step2-bspline
    2. Build a weighted transformation averaging all these transforms
    3. Invert the weighted transformation using step2-bspline-inversion
    4. Apply inverted weighted transform to volume i to move it to midpoint space

    This ensures all volumes are brought to the center of the transformation space.
    """
    import shutil

    n_vols = len(volumes)
    if n_vols == 1:
        return volumes

    po_step2 = itk.ParameterObject.New()
    po_step2.AddParameterFile(str(param_step2))

    po_inversion = itk.ParameterObject.New()
    po_inversion.AddParameterFile(str(param_step2_inversion))

    registered_volumes = []

    # Prepare identity transform file
    ref_itk = read_image_itk(volumes[0])
    identity_path = Path(temp_dir) / "identity.txt"
    write_identity_transform(identity_path, ref_itk)

    # For each volume i, bring it to midpoint space
    for i in range(n_vols):
        fixed_vol = volumes[i]
        fixed_itk = read_image_itk(fixed_vol)

        # Step 1: Register all volumes to volume i and collect transforms
        transform_paths = []

        for j in range(n_vols):
            if i == j:
                # Volume registering to itself = identity
                transform_paths.append(str(identity_path))
            else:
                # Register volume j to volume i with elastix writing to a temp directory
                moving_vol = volumes[j]
                moving_itk = read_image_itk(moving_vol)

                reg_output = Path(temp_dir) / f"reg_i{i}_j{j}"
                reg_output.mkdir(exist_ok=True)

                _, _ = itk.elastix_registration_method(
                    fixed_itk, moving_itk, parameter_object=po_step2, 
                    log_to_console=False, output_directory=str(reg_output)
                )

                # elastix writes TransformParameters.0.txt to the output directory
                transform_path = Path(temp_dir) / f"transform_i{i}_j{j}.txt"
                elastix_transform = reg_output / "TransformParameters.0.txt"
                if elastix_transform.exists():
                    shutil.copy(elastix_transform, transform_path)
                    shutil.rmtree(reg_output)
                else:
                    raise FileNotFoundError(f"Expected {elastix_transform} not found")

                transform_paths.append(str(transform_path))

        # Step 2: Build weighted combination (average of all transforms)
        weights = [1.0 / n_vols] * n_vols
        weighted_path = Path(temp_dir) / f"weighted_i{i}.txt"
        write_weighted_combination(
            weighted_path,
            weights=weights,
            subtransforms=transform_paths,
            image=fixed_itk,
        )

        # Step 3: Invert the weighted transformation
        po_weighted = itk.ParameterObject.New()
        po_weighted.AddParameterFile(str(weighted_path))

        inv_output = Path(temp_dir) / f"inv_i{i}"
        inv_output.mkdir(exist_ok=True)

        _, _ = itk.elastix_registration_method(
            fixed_itk, fixed_itk, initial_transform_parameter_file_name=str(weighted_path),
            parameter_object=po_inversion, log_to_console=False, output_directory=str(inv_output)
        )

        inverted_path = Path(temp_dir) / f"inverted_i{i}.txt"
        elastix_inverted = inv_output / "TransformParameters.0.txt"
        if elastix_inverted.exists():
            shutil.copy(elastix_inverted, inverted_path)
            shutil.rmtree(inv_output)
        else:
            raise FileNotFoundError(f"Expected {elastix_inverted} not found")

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            label = labels[i] if labels is not None else f"vol_{i}"
            final_path = output_dir / f"{label}_step2_final.txt"
            shutil.copy(inverted_path, final_path)

        # Step 4: Apply inverted weighted transform to volume i
        po_apply_inverted = itk.ParameterObject.New()
        po_apply_inverted.AddParameterFile(str(inverted_path))

        midpoint_img = itk.transformix_filter(
            fixed_itk, transform_parameter_object=po_apply_inverted, log_to_console=False
        )
        registered_volumes.append(itk.array_from_image(midpoint_img))

    return registered_volumes


def register_to_b0(volume, reference, param_step3, output_dir: Path = None, label: str = None):
    """
    Register a volume to the b=0 reference (Step 3 of Par0029).
    
    Parameters
    ----------
    volume : np.ndarray
        3D volume to register
    reference : np.ndarray
        3D reference volume (b=0)
    param_step3 : Path
        Path to step3 parameter file
        
    Returns
    -------
    registered_volume : np.ndarray
        Registered 3D volume
    """
    fixed_itk = read_image_itk(reference)
    moving_itk = read_image_itk(volume)
    
    po = itk.ParameterObject.New()
    po.AddParameterFile(str(param_step3))
    
    with tempfile.TemporaryDirectory() as temp_dir:
        kwargs = {
            "parameter_object": po,
            "log_to_console": False,
            "output_directory": temp_dir,
        }
        result_img, _ = itk.elastix_registration_method(
            fixed_itk, moving_itk, **kwargs
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            src = Path(temp_dir) / "TransformParameters.0.txt"
            if src.is_file():
                name = label or "volume"
                dst = output_dir / f"{name}_step3_final.txt"
                shutil.copy(src, dst)
    
    return itk.array_from_image(result_img)


def create_output_folder(subject_id, organ, direction, acquisition, test_type):
    """
    Create output folder structure.
    
    Format: {subject_id}_{organ}_{direction}_{acquisition}_{test_type}
    
    Parameters
    ----------
    subject_id : str
        Subject ID
    organ : str
        'liver' or 'lung'
    direction : str
        'axial' or 'coronal'
    acquisition : str
        'FB' or 'NT'
    test_type : str
        'test' or 'retest'
    
    Returns
    -------
    folder_path : str
        Full path to created folder
    """
    # Map direction names
    dir_map = {'axial': 'ax', 'coronal': 'cor'}
    dir_short = dir_map.get(direction, direction[:2])
    
    folder_name = f"{subject_id}_{organ}_{dir_short}_{acquisition}_{test_type}"
    folder_path = os.path.join(ROOT_OUTPUT_PATH, folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def process_scan(subject_id, organ, direction, acquisition, test_type, rawdata_path, rawdata_name, cohort_type, timing_results):
    """
    Process a single scan: reconstruct, register using Par0029, and save.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    organ : str
        'liver' or 'lung'
    direction : str
        'axial' or 'coronal'
    acquisition : str
        'FB' or 'NT' (free-breathing or navigator-triggered)
    test_type : str
        'test' or 'retest'
    rawdata_path : str
        Base path to raw data (from YAML)
    rawdata_name : str
        Raw data folder name
    cohort_type : str
        'volunteer' or 'patient'
    timing_results : dict
        Dictionary to store timing results
    """
    
    if not os.path.exists(rawdata_path):
        raise ValueError(f"  Warning: Recon data path not found: {rawdata_path}")
    
    # Check for required files in current path
    file_suffixes = ['.sin', '_imageIndex.txt', '_images.txt']
    required_files = [os.path.join(rawdata_path, rawdata_name + suffix) for suffix in file_suffixes]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    # If files are missing, try to find and copy them from parent directory
    if missing_files:
        import shutil
        parent_path = os.path.dirname(rawdata_path)
        still_missing = []
        
        for missing_file in missing_files:
            filename = os.path.basename(missing_file)
            parent_file = os.path.join(parent_path, filename)
            
            if os.path.exists(parent_file):
                # Copy file from parent to current path
                try:
                    shutil.copyfile(parent_file, missing_file)
                except (PermissionError, OSError) as e:
                    print(f"    Warning: Could not copy {filename}: {e}")
                    still_missing.append(missing_file)
                    continue
                
                # Verify copy was successful (separate from copy operation)
                if os.path.exists(missing_file):
                    print(f"    Copied {filename} from parent directory")
                else:
                    print(f"    Warning: Copy of {filename} failed - file not found after copy")
                    still_missing.append(missing_file)
            else:
                still_missing.append(missing_file)
        
        if still_missing:
            raise ValueError(f"  ERROR: Missing required files for {rawdata_name}:\n"
                           f"    Not found in {rawdata_path}\n"
                           f"    Not found in {parent_path}\n"
                           f"  Missing: {[os.path.basename(f) for f in still_missing]}")
    
    
    print(f"  {direction.upper()} {acquisition}: {rawdata_name}")
    
    # Set parameters based on direction and cohort type
    data_order = 0 if direction.lower() == 'axial' else 1
    
    # Step 1: Import and load reconstructed images
    print(f"    Loading data...")
    raw_data_full_path = os.path.join(rawdata_path, rawdata_name)

    try:
        [imDWI, unique_bvals] = import_images(raw_data_full_path, False, False, True, data_order)
        print(f"    Data shape: {imDWI.shape}, B-values: {unique_bvals}")
    except Exception as e:
        print(f"    Error during loading: {e}")
        return
   
    dirs_per_b = [0, 3, 3, 3] if len(unique_bvals) == 4 else [0, 3, 3]

    # Derive NSA per b-value directly from the acquired b_vals list
    [_, b_vals] = import_images(raw_data_full_path, True, False, False, data_order)
    b_vals_array = np.array(b_vals)
    vals, counts = np.unique(b_vals_array, return_counts=True)
    count_map = {float(v): int(c) for v, c in zip(vals, counts)}
    nsa_per_b = [count_map.get(float(bv), 0) for bv in unique_bvals]
    if any(nsa == 0 for nsa in nsa_per_b):
        raise ValueError(f"Unable to derive NSA for all b-values. unique_bvals={unique_bvals}, counts={count_map}")
    
    # Create output folder
    output_folder = create_output_folder(subject_id, organ, direction, acquisition, test_type)
    
    # Check if ADC_avg already exists
    adc_avg_filename = "adc_avg.nii"
    adc_avg_path = os.path.join(output_folder, adc_avg_filename)
    skip_avg_calculation = os.path.exists(adc_avg_path)
    
    # Step 2: Register volumes using Par0029 groupwise registration
    print(f"    Registering volumes (Par0029 Step 2 + Step 3)...")
    start_registration = time.time()
    
    dimX, dimY, dimZ, dimB, dimN, dimD = imDWI.shape
    midpoint_averaged = np.zeros((dimX, dimY, dimZ, dimB, dimD))
    trace_registered = np.zeros((dimX, dimY, dimZ, dimB))
    
    transforms_dir = Path(output_folder) / "transforms"
    step2_transforms_dir = transforms_dir / "step2"
    step3_transforms_dir = transforms_dir / "step3"
    step2_transforms_dir.mkdir(parents=True, exist_ok=True)
    step3_transforms_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Process each b-value
        for b in range(dimB):
            print(f"    Processing b={unique_bvals[b]}...")
            
            # Collect all volumes for this b-value
            if b == 0:
                bval_volumes = [imDWI[:, :, :, 0, n, 0] 
                    for n in range(nsa_per_b[b])]
                
                n_volumes = len(bval_volumes)
                print(f"      Number of volumes: {n_volumes}")
                
                # Step 2: Register to midpoint space
                print(f"      Step 2: Registering to midpoint...")
                labels = [f"b{unique_bvals[b]}_n{n}" for n in range(nsa_per_b[b])]
                midpoint_volumes = register_to_midpoint(
                    bval_volumes,
                    PARAM_STEP2_BSPLINE,
                    PARAM_STEP2_INVERSION,
                    temp_path,
                    output_dir=step2_transforms_dir,
                    labels=labels,
                )
            
                midpoint_averaged[:, :, :, b, 0] = np.mean(midpoint_volumes, axis=0)

            else:
                for d in range(dirs_per_b[b]):
                    bval_volumes = [imDWI[:, :, :, b, n, d] for n in range(4)] # Use only first 4 volumes for registration to avoid heavy computation burden (>1h)
                                    #for n in range(nsa_per_b[b])]

                    n_volumes = 4 #len(bval_volumes)
                    print(f"      Number of volumes: {n_volumes} (dir {d})")

                    # Step 2: Register to midpoint space
                    print(f"      Step 2: Registering to midpoint (dir {d})...")
                    labels = [f"b{unique_bvals[b]}_dir{d}_n{n}" for n in range(4)] # Use only first 4 volumes for registration to avoid heavy computation burden (>1h)
                    midpoint_volumes = register_to_midpoint(
                        bval_volumes,
                        PARAM_STEP2_BSPLINE,
                        PARAM_STEP2_INVERSION,
                        temp_path,
                        output_dir=step2_transforms_dir,
                        labels=labels,
                    )

                    midpoint_averaged[:, :, :, b, d] = np.mean(midpoint_volumes, axis=0)
            
        # Step 3: Register all midpoint volumes to b=0
        print(f"      Step 3: Registering to b=0...")
        for b in range(dimB):
            if b == 0:
                trace_registered[:, :, :, b] = midpoint_averaged[:, :, :, b, 0]
                identity_path = step3_transforms_dir / "b0_identity_step3_final.txt"
                if not identity_path.exists():
                    write_identity_transform(identity_path, read_image_itk(midpoint_averaged[:, :, :, b, 0]))
            else:
                registered_vols = []
                ref_vol = midpoint_averaged[:, :, :, 0, 0]
                for d in range(dirs_per_b[b]):
                    vol = midpoint_averaged[:, :, :, b, d]
                    if vol.shape != ref_vol.shape:
                        raise ValueError(f"Volume shape mismatch for b={unique_bvals[b]}, d={d}: "
                                         f"vol shape {vol.shape}, ref shape {ref_vol.shape}")
                    
                    label = f"b{unique_bvals[b]}_dir{d}"
                    registered_vol = register_to_b0(
                        vol,
                        ref_vol,
                        PARAM_STEP3_BSPLINE,
                        output_dir=step3_transforms_dir,
                        label=label,
                    )
                    registered_vols.append(registered_vol)

                trace_registered[:, :, :, b] = np.mean(registered_vols, axis=0)
    
    registration_time = time.time() - start_registration
    print(f"    Total registration time: {registration_time:.2f}s")

    # Step 3: Generate ADC map
    print(f"    Generating ADC map...")
    adc_registered = generate_ADC_standalone(trace_registered, unique_bvals, bmin=150, bmax=1000)
    
    # Step 4: Save outputs
    print(f"    Saving outputs...")
    
    # Save registered trace
    trace_filename = "trace_groupwise.nii"
    trace_filepath = os.path.join(output_folder, trace_filename)
    trace_nib = nib.Nifti1Image(trace_registered, np.eye(4))
    nib.save(trace_nib, trace_filepath)
    print(f"    Saved: {trace_filename}")
    
    # Save registered ADC
    adc_filename = "adc_groupwise.nii"
    adc_filepath = os.path.join(output_folder, adc_filename)
    adc_nib = nib.Nifti1Image(adc_registered[0], np.eye(4))
    nib.save(adc_nib, adc_filepath)
    print(f"    Saved: {adc_filename}")
    
    # Record timing
    timing_key = f"{subject_id}_{direction}_{acquisition}_{test_type}"
    timing_results[timing_key] = {
        'dirs_per_b': dirs_per_b,
        'nsa_per_b': nsa_per_b,
        'registration_time': registration_time,
        'output_path': trace_filepath
    }


def main():
    """Main processing function."""
    print("Starting DWI Par0029 Groupwise Registration Pipeline")
    print(f"Configuration: {DATA_LIST_PATH}")
    print(f"Output root: {ROOT_OUTPUT_PATH}")
    
    timing_results = {}
    
    # Process volunteers
    if 'volunteers' in config:
        print("\n" + "="*70)
        print("PROCESSING VOLUNTEERS")
        print("="*70)
        volunteers = config['volunteers']
        
        for subject_id, subject_data in volunteers.items():
            print(f"\n{subject_id}")
            organ = subject_data.get('organs', 'liver')
            
            # Process test and retest sessions
            for test_type in ['test', 'retest']:
                if test_type not in subject_data:
                    continue
                
                session = subject_data[test_type]
                rawdata_path = to_wsl_path(session.get('rawdata_path'))
                scans = session.get('scans', {})
                
                print(f"  {test_type.upper()} (session: {session.get('session_date')})")
                
                cohort_type = subject_data.get('cohort_type', 'volunteer')
                for direction, acq_dict in scans.items():
                    for acquisition, rawdata_name in acq_dict.items():
                        process_scan(subject_id, organ, direction, acquisition, test_type,
                                   rawdata_path, rawdata_name, cohort_type, timing_results)
    
    # Process patients
    if 'patients' in config:
        print("\n" + "="*70)
        print("PROCESSING PATIENTS")
        print("="*70)
        patients = config['patients']
        
        for subject_id, subject_data in patients.items():
            print(f"\n{subject_id}")
            organ = subject_data.get('organs', 'liver')
            
            # Process test and retest sessions
            for test_type in ['test', 'retest']:
                if test_type not in subject_data:
                    continue
                
                session = subject_data[test_type]
                rawdata_path = to_wsl_path(session.get('rawdata_path'))
                scans = session.get('scans', {})
                
                print(f"  {test_type.upper()} (session: {session.get('session_date')})")
                
                cohort_type = subject_data.get('cohort_type', 'patient')
                for direction, acq_dict in scans.items():
                    for acquisition, rawdata_name in acq_dict.items():
                        process_scan(subject_id, organ, direction, acquisition, test_type,
                                   rawdata_path, rawdata_name, cohort_type, timing_results)
    
    # Save timing results with compact array formatting
    timing_file = os.path.join(ROOT_OUTPUT_PATH, 'timing_results.json')
    with open(timing_file, 'w') as f:
        json.dump(timing_results, f, indent=2, cls=CompactJSONEncoder)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Timing results saved: {timing_file}")
    print("="*70)


if __name__ == '__main__':
    main()
