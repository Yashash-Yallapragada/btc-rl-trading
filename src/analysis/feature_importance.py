# Cell 15: Feature Importance Analysis
# Purpose: Identify most predictive indicators
# AI Coding Focus: SHAP values, permutation importance
# Expected Output: Ranked feature importance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import shap
import os
from datetime import datetime
import json
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Import environment
from updated_rl_env import BitcoinTradingEnv
from environment_setup import SEED

print("=" * 80)
print("CELL 15: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# ===========================
# 1. LOAD DATA AND MODELS
# ===========================
print("Loading data and trained models...")

# Load processed data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train_clean.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

print(f"Data shapes: X_train {X_train.shape}, X_test {X_test.shape}")

# Feature names (assuming these are your technical indicators)
feature_names = [
    'sma_10', 'sma_20', 'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'volume_sma', 'price_change', 'volatility',
    'momentum', 'stoch_k', 'stoch_d', 'williams_r', 'atr',
    'cci', 'roc', 'trix', 'dpo', 'kst'
]

# Adjust feature names to match actual number of features
if X_train.ndim == 3:
    actual_features = X_train.shape[2]
    X_train_2d = X_train[:, -1, :]  # Take last timestep for analysis
    X_test_2d = X_test[:, -1, :]
else:
    actual_features = X_train.shape[1]
    X_train_2d = X_train
    X_test_2d = X_test

# Adjust feature names to actual number
if len(feature_names) > actual_features:
    feature_names = feature_names[:actual_features]
elif len(feature_names) < actual_features:
    for i in range(len(feature_names), actual_features):
        feature_names.append(f'feature_{i}')

print(f"Using {len(feature_names)} features: {feature_names}")

# Load best RL model
model_paths = [
    'models/ppo_optimized_hyperparameters',
    'models/monitored/ppo_final_monitored',
    'models/ppo_bitcoin_optimized_final'
]

rl_model = None
for path in model_paths:
    try:
        rl_model = PPO.load(path)
        print(f"Loaded RL model from: {path}")
        break
    except:
        continue

if rl_model is None:
    print("Warning: Could not load RL model, will skip RL-specific analysis")

# ===========================
# 2. SURROGATE MODEL FOR FEATURE IMPORTANCE
# ===========================
print("\nCreating surrogate model for feature importance analysis...")

# Since direct RL feature importance is complex, we'll use surrogate approaches
print("Training Random Forest surrogate for price prediction...")

# Train RF surrogate model
rf_surrogate = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=SEED,
    n_jobs=-1
)

rf_surrogate.fit(X_train_2d, y_train)
rf_score = rf_surrogate.score(X_test_2d, y_test)
print(f"Random Forest R² score: {rf_score:.4f}")

# ===========================
# 3. TRADITIONAL FEATURE IMPORTANCE
# ===========================
print("\nCalculating traditional feature importance...")

# Random Forest feature importance
rf_importance = rf_surrogate.feature_importances_
rf_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

print("Top 10 features by Random Forest importance:")
print(rf_importance_df.head(10))

# ===========================
# 4. PERMUTATION IMPORTANCE
# ===========================
print("\nCalculating permutation importance...")

# Permutation importance for price prediction
perm_importance = permutation_importance(
    rf_surrogate, X_test_2d, y_test,
    n_repeats=5,
    random_state=SEED,
    n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Top 10 features by permutation importance:")
print(perm_importance_df.head(10))

# ===========================
# 5. SHAP ANALYSIS (SIMPLIFIED)
# ===========================
print("\nCalculating SHAP values...")

try:
    # Use subset of data for SHAP (computational efficiency)
    shap_sample_size = min(100, len(X_test_2d))
    X_shap = X_test_2d[:shap_sample_size]
    
    print(f"Computing SHAP values for {shap_sample_size} samples...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf_surrogate)
    shap_values = explainer.shap_values(X_shap)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(0)
    
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_shap
    }).sort_values('shap_importance', ascending=False)
    
    print("Top 10 features by SHAP importance:")
    print(shap_importance_df.head(10))
    
    shap_success = True
    
except Exception as e:
    print(f"SHAP analysis failed: {e}")
    print("Continuing with other importance methods...")
    shap_success = False
    shap_importance_df = pd.DataFrame()

# ===========================
# 6. RL AGENT ACTION SENSITIVITY ANALYSIS
# ===========================
print("\nAnalyzing RL agent sensitivity to features...")

rl_feature_importance = []

if rl_model is not None:
    try:
        def make_eval_env(X_data, y_data):
            def _init():
                if X_data.ndim == 3:
                    X_data_2d = X_data[:, -1, :]
                else:
                    X_data_2d = X_data
                    
                env = BitcoinTradingEnv(
                    X_data=X_data_2d,
                    y_data=y_data,
                    window_size=60,
                    initial_balance=100000,
                    mode="eval"
                )
                return Monitor(env)
            return _init

        # Create baseline environment
        baseline_env = DummyVecEnv([make_eval_env(X_test, y_test)])
        
        # Get baseline performance
        baseline_returns = []
        obs = baseline_env.reset()
        
        for _ in range(min(200, len(y_test)-10)):
            action, _ = rl_model.predict(obs, deterministic=True)
            obs, reward, done, info = baseline_env.step(action)
            baseline_returns.append(reward[0])
            if done[0]:
                break
                
        baseline_performance = np.mean(baseline_returns)
        
        print(f"Baseline RL performance: {baseline_performance:.4f}")
        
        # Test feature importance by perturbation
        for i, feature_name in enumerate(feature_names[:10]):  # Test top 10 features only
            print(f"Testing feature {i+1}/10: {feature_name}")
            
            # Create perturbed data
            X_test_perturbed = X_test.copy()
            if X_test_perturbed.ndim == 3:
                # Perturb feature across all timesteps
                X_test_perturbed[:, :, i] = np.random.permutation(X_test_perturbed[:, :, i])
            else:
                X_test_perturbed[:, i] = np.random.permutation(X_test_perturbed[:, i])
            
            # Test with perturbed data
            perturbed_env = DummyVecEnv([make_eval_env(X_test_perturbed, y_test)])
            perturbed_returns = []
            obs = perturbed_env.reset()
            
            for _ in range(min(200, len(y_test)-10)):
                action, _ = rl_model.predict(obs, deterministic=True)
                obs, reward, done, info = perturbed_env.step(action)
                perturbed_returns.append(reward[0])
                if done[0]:
                    break
                    
            perturbed_performance = np.mean(perturbed_returns)
            
            # Calculate importance as performance drop
            importance = baseline_performance - perturbed_performance
            rl_feature_importance.append({
                'feature': feature_name,
                'rl_importance': importance,
                'baseline_perf': baseline_performance,
                'perturbed_perf': perturbed_performance
            })
            
            perturbed_env.close()
            
        baseline_env.close()
        
        # Convert to DataFrame
        rl_importance_df = pd.DataFrame(rl_feature_importance).sort_values('rl_importance', ascending=False)
        
        print("Top 10 features by RL sensitivity:")
        print(rl_importance_df.head(10))
        
        rl_analysis_success = True
        
    except Exception as e:
        print(f"RL feature importance analysis failed: {e}")
        rl_analysis_success = False
        rl_importance_df = pd.DataFrame()
else:
    rl_analysis_success = False
    rl_importance_df = pd.DataFrame()

# ===========================
# 7. COMBINE AND RANK FEATURE IMPORTANCE
# ===========================
print("\nCombining feature importance rankings...")

# Combine all importance measures
combined_importance = pd.DataFrame({'feature': feature_names})

# Add RF importance
combined_importance = combined_importance.merge(
    rf_importance_df[['feature', 'importance']].rename(columns={'importance': 'rf_importance'}),
    on='feature', how='left'
)

# Add permutation importance
combined_importance = combined_importance.merge(
    perm_importance_df[['feature', 'importance_mean']].rename(columns={'importance_mean': 'perm_importance'}),
    on='feature', how='left'
)

# Add SHAP importance if available
if shap_success:
    combined_importance = combined_importance.merge(
        shap_importance_df[['feature', 'shap_importance']],
        on='feature', how='left'
    )

# Add RL importance if available
if rl_analysis_success:
    combined_importance = combined_importance.merge(
        rl_importance_df[['feature', 'rl_importance']],
        on='feature', how='left'
    )

# Calculate composite importance score
importance_cols = ['rf_importance', 'perm_importance']
if shap_success:
    importance_cols.append('shap_importance')
if rl_analysis_success:
    importance_cols.append('rl_importance')

# Normalize each importance measure to 0-1 scale
for col in importance_cols:
    if col in combined_importance.columns:
        min_val = combined_importance[col].min()
        max_val = combined_importance[col].max()
        if max_val > min_val:
            combined_importance[f'{col}_norm'] = (combined_importance[col] - min_val) / (max_val - min_val)
        else:
            combined_importance[f'{col}_norm'] = 0

# Calculate composite score (average of normalized importance)
norm_cols = [f'{col}_norm' for col in importance_cols if f'{col}_norm' in combined_importance.columns]
combined_importance['composite_importance'] = combined_importance[norm_cols].mean(axis=1)

# Sort by composite importance
final_importance = combined_importance.sort_values('composite_importance', ascending=False)

print("\nFinal feature importance ranking:")
print(final_importance[['feature', 'composite_importance'] + importance_cols].head(15))

# ===========================
# 8. FEATURE SELECTION AND REDUCED DATASET
# ===========================
print("\nSelecting top features and creating reduced dataset...")

# Select top N features
top_n_features = 10
top_features = final_importance.head(top_n_features)['feature'].tolist()

print(f"Selected top {top_n_features} features:")
for i, feature in enumerate(top_features, 1):
    importance_score = final_importance[final_importance['feature'] == feature]['composite_importance'].iloc[0]
    print(f"  {i}. {feature}: {importance_score:.4f}")

# Create reduced datasets
feature_indices = [feature_names.index(f) for f in top_features]

if X_train.ndim == 3:
    X_train_reduced = X_train[:, :, feature_indices]
    X_test_reduced = X_test[:, :, feature_indices]
else:
    X_train_reduced = X_train[:, feature_indices]
    X_test_reduced = X_test[:, feature_indices]

print(f"Reduced dataset shapes: X_train {X_train_reduced.shape}, X_test {X_test_reduced.shape}")

# Save reduced datasets
os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train_reduced.npy', X_train_reduced)
np.save('data/processed/X_test_reduced.npy', X_test_reduced)
np.save('data/processed/y_train_reduced.npy', y_train)
np.save('data/processed/y_test_reduced.npy', y_test)

print("Reduced datasets saved to data/processed/")

# ===========================
# 9. VISUALIZATIONS
# ===========================
print("\nCreating feature importance visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Feature Importance Analysis Results', fontsize=16, fontweight='bold')

# Plot 1: Composite importance (top 15 features)
ax1 = axes[0, 0]
top_15 = final_importance.head(15)
bars = ax1.barh(range(len(top_15)), top_15['composite_importance'], color='steelblue')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['feature'])
ax1.set_xlabel('Composite Importance Score')
ax1.set_title('Top 15 Features - Composite Importance')
ax1.grid(True, alpha=0.3)

# Add values to bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center')

# Plot 2: Comparison of importance methods
ax2 = axes[0, 1]
top_10_for_comparison = final_importance.head(10)
x = np.arange(len(top_10_for_comparison))
width = 0.25

methods = []
if 'rf_importance' in top_10_for_comparison.columns:
    methods.append(('RF', 'rf_importance', 'blue'))
if 'perm_importance' in top_10_for_comparison.columns:
    methods.append(('Perm', 'perm_importance', 'green'))
if 'shap_importance' in top_10_for_comparison.columns:
    methods.append(('SHAP', 'shap_importance', 'red'))

for i, (label, col, color) in enumerate(methods):
    values = top_10_for_comparison[f'{col}_norm']
    ax2.bar(x + i * width, values, width, label=label, color=color, alpha=0.7)

ax2.set_xlabel('Features')
ax2.set_ylabel('Normalized Importance')
ax2.set_title('Importance Methods Comparison (Top 10)')
ax2.set_xticks(x + width)
ax2.set_xticklabels(top_10_for_comparison['feature'], rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Feature importance distribution
ax3 = axes[1, 0]
ax3.hist(final_importance['composite_importance'], bins=15, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(final_importance['composite_importance'].mean(), color='red', linestyle='--', 
           label=f'Mean: {final_importance["composite_importance"].mean():.3f}')
ax3.set_xlabel('Composite Importance Score')
ax3.set_ylabel('Number of Features')
ax3.set_title('Distribution of Feature Importance Scores')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Cumulative importance
ax4 = axes[1, 1]
cumulative_importance = np.cumsum(final_importance['composite_importance'])
normalized_cumulative = cumulative_importance / cumulative_importance.iloc[-1]

ax4.plot(range(1, len(normalized_cumulative)+1), normalized_cumulative, 'o-', color='darkgreen')
ax4.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='80% threshold')
ax4.axhline(0.9, color='orange', linestyle='--', alpha=0.7, label='90% threshold')

# Find number of features for 80% and 90% importance
features_80 = np.argmax(normalized_cumulative >= 0.8) + 1
features_90 = np.argmax(normalized_cumulative >= 0.9) + 1

ax4.axvline(features_80, color='red', linestyle=':', alpha=0.7)
ax4.axvline(features_90, color='orange', linestyle=':', alpha=0.7)

ax4.set_xlabel('Number of Features')
ax4.set_ylabel('Cumulative Importance (Normalized)')
ax4.set_title('Cumulative Feature Importance')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===========================
# 10. SAVE RESULTS
# ===========================
print("\nSaving feature importance analysis results...")

# Compile comprehensive results
feature_analysis_results = {
    'analysis_info': {
        'total_features': len(feature_names),
        'top_features_selected': top_n_features,
        'methods_used': importance_cols,
        'shap_analysis_success': shap_success,
        'rl_analysis_success': rl_analysis_success,
        'analysis_date': datetime.now().isoformat()
    },
    'feature_rankings': {
        'top_features': top_features,
        'complete_ranking': final_importance.to_dict('records')
    },
    'importance_scores': {
        'rf_surrogate_score': rf_score,
        'features_for_80_percent': int(features_80),
        'features_for_90_percent': int(features_90)
    },
    'reduced_dataset_info': {
        'original_shape': X_train.shape,
        'reduced_shape': X_train_reduced.shape,
        'selected_features': top_features,
        'reduction_ratio': X_train_reduced.shape[-1] / X_train.shape[-1]
    }
}

# Save results
os.makedirs('results', exist_ok=True)

with open('results/feature_importance_analysis.json', 'w') as f:
    json.dump(feature_analysis_results, f, indent=2, default=str)

with open('results/feature_importance_analysis.pkl', 'wb') as f:
    pickle.dump(feature_analysis_results, f)

# Save detailed dataframes
final_importance.to_csv('results/feature_importance_detailed.csv', index=False)

print("Feature importance analysis results saved")

# ===========================
# 11. PERFORMANCE COMPARISON
# ===========================
print("\nComparing performance with reduced feature set...")

try:
    # Train RF on reduced features
    rf_reduced = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=SEED,
        n_jobs=-1
    )
    
    if X_train.ndim == 3:
        X_train_2d_reduced = X_train_reduced[:, -1, :]
        X_test_2d_reduced = X_test_reduced[:, -1, :]
    else:
        X_train_2d_reduced = X_train_reduced
        X_test_2d_reduced = X_test_reduced
    
    rf_reduced.fit(X_train_2d_reduced, y_train)
    rf_reduced_score = rf_reduced.score(X_test_2d_reduced, y_test)
    
    print(f"Performance comparison:")
    print(f"- Full feature set R²: {rf_score:.4f}")
    print(f"- Reduced feature set R²: {rf_reduced_score:.4f}")
    print(f"- Performance retention: {(rf_reduced_score/rf_score)*100:.1f}%")
    print(f"- Feature reduction: {len(feature_names)} → {top_n_features} ({(top_n_features/len(feature_names))*100:.1f}%)")
    
    performance_retained = rf_reduced_score >= (rf_score * 0.95)  # 95% threshold
    
except Exception as e:
    print(f"Performance comparison failed: {e}")
    performance_retained = False

# ===========================
# 12. FINAL REPORT
# ===========================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS FINAL REPORT")
print("=" * 80)

print(f"ANALYSIS SUMMARY:")
print(f"- Total features analyzed: {len(feature_names)}")
print(f"- Feature importance methods used: {len(importance_cols)}")
print(f"- Top features selected: {top_n_features}")

print(f"\nTOP 5 MOST IMPORTANT FEATURES:")
for i, feature in enumerate(top_features[:5], 1):
    importance_score = final_importance[final_importance['feature'] == feature]['composite_importance'].iloc[0]
    print(f"  {i}. {feature}: {importance_score:.4f}")

print(f"\nMETHODS SUCCESSFULLY APPLIED:")
print(f"- Random Forest importance: ✓")
print(f"- Permutation importance: ✓")
print(f"- SHAP analysis: {'✓' if shap_success else '✗'}")
print(f"- RL agent sensitivity: {'✓' if rl_analysis_success else '✗'}")

print(f"\nFEATURE SELECTION RESULTS:")
print(f"- Features needed for 80% importance: {features_80}")
print(f"- Features needed for 90% importance: {features_90}")
print(f"- Recommended feature set size: {top_n_features}")

print(f"\nDATASET REDUCTION:")
print(f"- Original features: {len(feature_names)}")
print(f"- Reduced features: {top_n_features}")
print(f"- Size reduction: {((len(feature_names)-top_n_features)/len(feature_names))*100:.1f}%")

if 'performance_retained' in locals():
    print(f"- Performance retained: {'Yes' if performance_retained else 'No'}")

print(f"\nCORE REQUIREMENTS FULFILLED:")
print(f"✓ Feature importance calculation using multiple methods")
print(f"{'✓' if shap_success else '○'} SHAP values computed")
print(f"✓ Permutation importance analysis")
print(f"{'✓' if rl_analysis_success else '○'} RL agent decision analysis")
print(f"✓ Feature ranking and selection")
print(f"✓ Reduced dataset creation")

print(f"\nFILES SAVED:")
print(f"- results/feature_importance_analysis.json")
print(f"- results/feature_importance_analysis.pkl")
print(f"- results/feature_importance_detailed.csv")
print(f"- results/feature_importance_analysis.png")
print(f"- data/processed/X_train_reduced.npy")
print(f"- data/processed/X_test_reduced.npy")

print(f"\nRECOMMENDATIONS:")
if performance_retained if 'performance_retained' in locals() else True:
    print(f"- Use reduced feature set for improved training efficiency")
    print(f"- Focus on top {min(5, top_n_features)} features for model interpretation")
else:
    print(f"- Consider increasing number of selected features")
    print(f"- Validate reduced model performance before deployment")

print("\n" + "=" * 80)
print("CELL 15 COMPLETE: Feature importance analysis finished")
print("=" * 80)