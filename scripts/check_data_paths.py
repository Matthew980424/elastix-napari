import os
import yaml

# Configuration
DATA_LIST_PATH = 'recon_data.yaml'

# Load configuration
with open(DATA_LIST_PATH, 'r') as f:
    config = yaml.safe_load(f)

def check_files(subject_id, test_type, rawdata_path, rawdata_name, direction, acquisition):
    """Check if required files exist for a scan, copy from parent if needed."""
    
    issues = []
    
    # Check if rawdata_path exists
    if not os.path.exists(rawdata_path):
        issues.append(f"    ✗ Path not found: {rawdata_path}")
        return issues
    else:
        print(f"    ✓ Path exists: {rawdata_path}")
    
    # Check for required files
    import shutil
    file_suffixes = ['.sin', '_imageIndex.txt', '_images.txt']
    required_files = [os.path.join(rawdata_path, rawdata_name + suffix) for suffix in file_suffixes]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    # If files are missing, try to copy from parent directory
    if missing_files:
        parent_path = os.path.dirname(rawdata_path)
        still_missing = []
        
        for missing_file in missing_files:
            filename = os.path.basename(missing_file)
            parent_file = os.path.join(parent_path, filename)
            
            if os.path.exists(parent_file):
                # Copy file from parent to current path
                try:
                    shutil.copyfile(parent_file, missing_file)
                    if os.path.exists(missing_file):
                        print(f"      ✓ Copied {filename} from parent directory")
                    else:
                        print(f"      ✗ Copy failed: {filename}")
                        still_missing.append(filename)
                except (PermissionError, OSError) as e:
                    print(f"      ✗ Could not copy {filename}: {e}")
                    still_missing.append(filename)
            else:
                print(f"      ✗ Missing: {filename} (not in current or parent path)")
                still_missing.append(filename)
        
        if still_missing:
            issues.append(f"    ✗ Missing files: {', '.join(still_missing)}")
    else:
        # All files found
        for req_file in required_files:
            filename = os.path.basename(req_file)
            print(f"      ✓ Found: {filename}")
    
    return issues

def main():
    """Check all data paths and files in the configuration."""
    print("="*70)
    print("DATA PATH AND FILE VALIDATION")
    print("="*70)
    print(f"Configuration file: {DATA_LIST_PATH}\n")
    
    all_issues = []
    total_scans = 0
    valid_scans = 0
    
    # Process volunteers
    if 'volunteers' in config:
        print("\n" + "="*70)
        print("VOLUNTEERS")
        print("="*70)
        volunteers = config['volunteers']
        
        for subject_id, subject_data in volunteers.items():
            print(f"\n{subject_id}")
            organ = subject_data.get('organs', 'liver')
            cohort_type = subject_data.get('cohort_type', 'volunteer')
            print(f"  Organ: {organ}, Cohort: {cohort_type}")
            
            # Process test and retest sessions
            for test_type in ['test', 'retest']:
                if test_type not in subject_data:
                    continue
                
                session = subject_data[test_type]
                rawdata_path = session.get('rawdata_path')
                scans = session.get('scans', {})
                
                print(f"\n  {test_type.upper()} (session: {session.get('session_date')})")
                
                for direction, acq_dict in scans.items():
                    for acquisition, rawdata_name in acq_dict.items():
                        total_scans += 1
                        print(f"\n  {direction.upper()} {acquisition}: {rawdata_name}")
                        
                        issues = check_files(subject_id, test_type, rawdata_path, 
                                           rawdata_name, direction, acquisition)
                        
                        if not issues:
                            valid_scans += 1
                            print(f"    ✓ All files OK")
                        else:
                            all_issues.extend([f"{subject_id}_{test_type}_{direction}_{acquisition}:"] + issues)
    
    # Process patients
    if 'patients' in config:
        print("\n" + "="*70)
        print("PATIENTS")
        print("="*70)
        patients = config['patients']
        
        for subject_id, subject_data in patients.items():
            print(f"\n{subject_id}")
            organ = subject_data.get('organs', 'liver')
            cohort_type = subject_data.get('cohort_type', 'patient')
            print(f"  Organ: {organ}, Cohort: {cohort_type}")
            
            # Process test and retest sessions
            for test_type in ['test', 'retest']:
                if test_type not in subject_data:
                    continue
                
                session = subject_data[test_type]
                rawdata_path = session.get('rawdata_path')
                scans = session.get('scans', {})
                
                print(f"\n  {test_type.upper()} (session: {session.get('session_date')})")
                
                for direction, acq_dict in scans.items():
                    for acquisition, rawdata_name in acq_dict.items():
                        total_scans += 1
                        print(f"\n  {direction.upper()} {acquisition}: {rawdata_name}")
                        
                        issues = check_files(subject_id, test_type, rawdata_path, 
                                           rawdata_name, direction, acquisition)
                        
                        if not issues:
                            valid_scans += 1
                            print(f"    ✓ All files OK")
                        else:
                            all_issues.extend([f"{subject_id}_{test_type}_{direction}_{acquisition}:"] + issues)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total scans checked: {total_scans}")
    print(f"Valid scans: {valid_scans}")
    print(f"Scans with issues: {total_scans - valid_scans}")
    
    if all_issues:
        print("\n" + "="*70)
        print("ISSUES FOUND")
        print("="*70)
        for issue in all_issues:
            print(issue)
    else:
        print("\n✓ All paths and files are valid!")

if __name__ == '__main__':
    main()
