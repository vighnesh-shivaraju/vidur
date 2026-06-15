#using speculative decoding
def run_iteration(self, batch):
    #defining speculative tokens 'K'
    K = self.config.speculative_window

    #time taken by draft model to guess k tokens
    draft_latency = self.draft_model_estimator.get_runtime(batch, num_tokens=K)

    #training all tokens in parallel rather than doing in individual steps
    verify_latency = self.runtime_estimator.get_runtime(batch, num_tokens=K, is_verify=True)

    #determining actual number of accpeted tokens
    accepted_count = self.simulation_policy.get_accepted_tokens(batch, K)

    #updating state with multiple tokens by adding multiple tokens
    for request in batch.requests:
        request.on_tokens_generated(accepted_count)

    #total time is being made sum of time taken by draft and target model
    total_latency = draft_latency + verify_latency
    self.metrics_tracker.on_iteration_complete(total_latency)
    return total_latency
