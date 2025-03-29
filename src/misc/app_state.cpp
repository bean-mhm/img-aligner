#include "app_state.hpp"

namespace img_aligner
{

    const bv::CommandPoolPtr& AppState::cmd_pool(bool transient)
    {
        if (!device)
        {
            throw std::runtime_error(
                "command pool requested before device creation"
            );
        }

        std::unordered_map<std::thread::id, bv::CommandPoolPtr>& pools =
            transient ? transient_cmd_pools : cmd_pools;

        auto thread_id = std::this_thread::get_id();
        if (!pools.contains(thread_id))
        {
            VkCommandPoolCreateFlags flags = 0;
            if (transient)
            {
                flags |= VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            }

            pools[thread_id] = bv::CommandPool::create(
                device,
                {
                    .flags = flags,
                    .queue_family_index = queue_main->queue_family_index()
                }
            );
        }
        return pools[thread_id];
    }

}
