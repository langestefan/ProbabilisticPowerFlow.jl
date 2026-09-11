module PPFOhMyThreadsExt

using OhMyThreads: Scheduler, chunks, tmapreduce
using ProbabilisticPowerFlow: ProbabilisticPowerFlow

function ProbabilisticPowerFlow.solve_blocks(f, scheduler::Scheduler, order)
    blocks = chunks(order; n = Threads.nthreads())
    return tmapreduce(f, vcat, blocks; scheduler = scheduler)
end

end
