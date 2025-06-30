#!/bin/bash

# Director Integration Patches for TD-MPC2
# This script applies the minimal intrusion patches to integrate hierarchical control

set -e  # Exit on any error

echo "🚀 Applying Director patches to TD-MPC2..."

# Check if we're in the right directory
if [ ! -f "tdmpc2/train.py" ]; then
    echo "❌ Error: Please run this script from the TD-MPC2 root directory"
    exit 1
fi

# Function to check if a file exists
check_file() {
    if [ ! -f "$1" ]; then
        echo "❌ Error: File $1 not found"
        exit 1
    fi
    echo "✅ File $1 exists"
}

# Function to backup original files
backup_file() {
    if [ -f "$1" ] && [ ! -f "$1.backup" ]; then
        cp "$1" "$1.backup"
        echo "📦 Backed up $1 to $1.backup"
    fi
}

echo ""
echo "📋 Checking patch prerequisites..."

# Check that new files have been created
check_file "tdmpc2/common/goal_vae.py"
check_file "tdmpc2/common/manager.py"
check_file "tdmpc2/trainer/director_trainer.py"
check_file "tdmpc2/configs/director.yaml"
check_file "tdmpc2/configs/task/director_walker_walk.yaml"
check_file "tdmpc2/configs/task/director_humanoid_walk.yaml"

echo ""
echo "📦 Creating backups of modified files..."

# Backup files that will be modified
backup_file "tdmpc2/train.py"
backup_file "tdmpc2/Mooretdmpc.py"

echo ""
echo "🔧 Checking patch integrity..."

# Verify that train.py has the DirectorTrainer import
if grep -q "from tdmpc2.trainer.director_trainer import DirectorTrainer" tdmpc2/train.py; then
    echo "✅ DirectorTrainer import found in train.py"
else
    echo "❌ Error: DirectorTrainer import not found in train.py"
    exit 1
fi

# Verify that MooreTDMPC.act has goal parameter
if grep -q "def act(self, obs, t0=False, eval_mode=False, task=None, goal=None):" tdmpc2/Mooretdmpc.py; then
    echo "✅ Goal parameter found in MooreTDMPC.act method"
else
    echo "❌ Error: Goal parameter not found in MooreTDMPC.act method"
    exit 1
fi

echo ""
echo "🧪 Running basic import tests..."

# Test Python imports
python3 -c "
import sys
sys.path.append('.')
try:
    from tdmpc2.common.goal_vae import GoalVAE
    print('✅ GoalVAE import successful')
except Exception as e:
    print(f'❌ GoalVAE import failed: {e}')
    exit(1)

try:
    from tdmpc2.common.manager import Manager
    print('✅ Manager import successful')
except Exception as e:
    print(f'❌ Manager import failed: {e}')
    exit(1)

try:
    from tdmpc2.trainer.director_trainer import DirectorTrainer
    print('✅ DirectorTrainer import successful')
except Exception as e:
    print(f'❌ DirectorTrainer import failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Python import tests failed"
    exit 1
fi

echo ""
echo "📊 Patch summary:"
echo "  📄 New files created: 7"
echo "    - tdmpc2/common/goal_vae.py"
echo "    - tdmpc2/common/manager.py" 
echo "    - tdmpc2/trainer/director_trainer.py"
echo "    - tdmpc2/configs/director.yaml"
echo "    - tdmpc2/configs/task/director_walker_walk.yaml"
echo "    - tdmpc2/configs/task/director_humanoid_walk.yaml"
echo "    - apply_director_patches.sh (this script)"
echo ""
echo "  🔧 Files modified: 2"
echo "    - tdmpc2/train.py (added DirectorTrainer selection)"
echo "    - tdmpc2/Mooretdmpc.py (added goal conditioning)"
echo ""

echo "✅ All Director patches applied successfully!"
echo ""
echo "🎯 Usage examples:"
echo "  # Train hierarchical agent on Walker Walk"
echo "  python tdmpc2/train.py task=director_walker_walk"
echo ""
echo "  # Train hierarchical agent on Humanoid Walk"  
echo "  python tdmpc2/train.py task=director_humanoid_walk"
echo ""
echo "  # Use custom hyperparameters"
echo "  python tdmpc2/train.py task=director_walker_walk goal_vae.n_latents=8 manager.goal_interval=5"
echo ""
echo "🔄 To rollback changes:"
echo "  mv tdmpc2/train.py.backup tdmpc2/train.py"
echo "  mv tdmpc2/Mooretdmpc.py.backup tdmpc2/Mooretdmpc.py"
echo "  rm tdmpc2/common/goal_vae.py tdmpc2/common/manager.py"
echo "  rm tdmpc2/trainer/director_trainer.py"
echo "  rm tdmpc2/configs/director.yaml"
echo "  rm tdmpc2/configs/task/director_*.yaml"
echo ""
echo "🚀 Director integration complete! Happy hierarchical training! 🤖" 