from client import GroupRelativePolicyOptimizationReinforcementLearningClient

def main():
    client = GroupRelativePolicyOptimizationReinforcementLearningClient()
    res = client.compute_grpo_step(8, 0.70)
    print('GRPO Policy Optimizer: ' + res['grpo_update_id'] + ' (Group Size: ' + str(res['group_size']) + ')')
    print('Critic Model Bypassed: ' + str(res['critic_model_bypassed']) + ' | Relative Advantage: +' + str(res['relative_advantage_gain']))
    print('KL Divergence: ' + str(res['kl_divergence_penalty']) + ' | Stabilized: ' + str(res['policy_entropy_stabilized']))

if __name__ == '__main__':
    main()
