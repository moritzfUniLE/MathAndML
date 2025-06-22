# Project Cleanup Summary

## Cleanup Dates
- Initial Cleanup: 2025-01-19
- Recent Cleanup: 2025-06-20

## Files Archived

### Recent Cleanup (2025-06-20)

#### Removed Files
- **BNLEARN_SETUP.md**: Separate bnlearn setup documentation
  - Content integrated into main README.md installation section
  - Eliminated redundant documentation
  - Simplified setup process for users

### Legacy Code (`archive/legacy_code/`)
- **notears_core.py**: Original monolithic implementation
  - Contains the old NotearsMLP class and notears_nonlinear function
  - Superseded by modular algorithm system in `algorithms/`
  - Utility functions extracted to `notears_utils.py`

- **notears_gui.py**: Failed desktop GUI attempt
  - Tkinter-based GUI that never worked due to missing tkinter module
  - Superseded by web-based GUI (`notears_web_gui.py`)

- **test_datasets.py**: Legacy testing framework
  - Old command-line testing script
  - Functionality integrated into web GUI interface

- **add_bnlearn_datasets.py**: Dataset integration script (Added 2025-06-20)
  - Script to download and integrate bnlearn datasets
  - No longer needed as datasets are included in repository
  - All bnlearn datasets now pre-loaded in `datasets/` directory

### Legacy Results (`archive/legacy_results/`)
- **test_results/**: Old test outputs and visualizations
  - Contains results from previous testing runs
  - JSON files, PNG images, and CSV outputs
  - Preserved for reference but no longer actively used

### Previously Archived (`archive/old_*`)
- **old_results/**: Even older result archives from previous cleanups
- **old_scripts/**: Legacy scripts from earlier development phases

## Current Project Structure

### Active Files
- `algorithms/`: New modular algorithm system
- `notears_web_gui.py`: Main web application
- `notears_utils.py`: Extracted utility functions
- `launch_gui.py`: Application launcher
- `create_test_datasets.py`: Dataset generator
- `templates/` & `static/`: Web GUI assets
- `datasets/`: Test datasets
- `requirements.txt`: Dependencies
- `README.md`: Updated documentation

### Key Improvements
1. **Modular Architecture**: Algorithms are now pluggable components
2. **Clean Separation**: Utilities separated from core algorithm logic
3. **Single Interface**: Web GUI is the primary user interface
4. **Better Organization**: Clear distinction between active and archived code

## Migration Notes

### Import Changes
- `from notears_core import` → `from notears_utils import` (for utilities)
- Algorithm functionality now accessed via `algorithms` module

### Functionality Preservation
- All utility functions preserved in `notears_utils.py`
- Both Linear and Nonlinear algorithms available via modular system
- Web GUI maintains all previous functionality plus algorithm selection

### Benefits
- Easier to add new algorithms
- Cleaner codebase with less duplication
- Better separation of concerns
- Preserved backward compatibility for datasets and results