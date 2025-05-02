// NOTE: this version of beva is modified to use the fmt library instead of
// std::format.

#include "beva.hpp"

// define a derived class named ClassName_public_ctor that lets us use the
// previously private constructors (actually protected, just go with it) as
// public ones so that they can be used in std::make_shared() or whatever
// else.
#define _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(ClassName) \
    class ClassName##_public_ctor : public ClassName \
    { \
    public: \
        template<typename... Args> ClassName##_public_ctor(Args&&... args) \
            : ClassName(std::forward<Args>(args)...) \
        {} \
    }

// round up integer division
#define _BV_IDIV_CEIL(a, b) (((a) + (b) - 1) / (b))

namespace bv
{

    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(PhysicalDevice);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Context);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(DebugMessenger);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Surface);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Queue);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Device);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Image);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Swapchain);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(ImageView);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(ShaderModule);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Sampler);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(DescriptorSetLayout);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(PipelineLayout);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(RenderPass);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(GraphicsPipeline);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(ComputePipeline);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Framebuffer);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(CommandBuffer);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(CommandPool);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Semaphore);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Fence);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(Buffer);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(DeviceMemory);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(DescriptorSet);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(DescriptorPool);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(BufferView);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(PipelineCache);

    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(MemoryRegion);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(MemoryChunk);
    _BV_DEFINE_DERIVED_WITH_PUBLIC_CONSTRUCTOR(MemoryBank);

#define _BV_LOCK_WPTR_OR_RETURN(wptr, locked_name) \
    if (wptr.expired()) \
    { \
        return; \
    } \
    auto locked_name = wptr.lock();

#pragma region forward declarations

    static void* vk_allocation_callback(
        void* p_user_data,
        size_t size,
        size_t alignment,
        VkSystemAllocationScope allocation_scope
    );
    static void* vk_reallocation_callback(
        void* p_user_data,
        void* p_original,
        size_t size,
        size_t alignment,
        VkSystemAllocationScope allocation_scope
    );
    static void vk_free_callback(
        void* p_user_data,
        void* p_memory
    );
    static void vk_internal_allocation_notification(
        void* p_user_data,
        size_t size,
        VkInternalAllocationType allocation_type,
        VkSystemAllocationScope allocation_scope
    );
    static void vk_internal_free_notification(
        void* p_user_data,
        size_t size,
        VkInternalAllocationType allocation_type,
        VkSystemAllocationScope allocation_scope
    );
    static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_types,
        const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
        void* p_user_data
    );

#pragma endregion

#pragma region Vulkan function loaders for extensions

    static VkResult CreateDebugUtilsMessengerEXT(
        VkInstance instance,
        const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
        const VkAllocationCallbacks* pAllocator,
        VkDebugUtilsMessengerEXT* pDebugMessenger
    )
    {
        auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance,
            "vkCreateDebugUtilsMessengerEXT"
        );
        if (func != nullptr)
        {
            return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
        }
        else
        {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    static void DestroyDebugUtilsMessengerEXT(
        VkInstance instance,
        VkDebugUtilsMessengerEXT debugMessenger,
        const VkAllocationCallbacks* pAllocator
    )
    {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance,
            "vkDestroyDebugUtilsMessengerEXT"
        );
        if (func != nullptr)
        {
            func(instance, debugMessenger, pAllocator);
        }
    }

#pragma endregion

#pragma region data-only structs and enums

    std::string Version::to_string() const
    {
        return fmt::format("{}.{}.{}.{}", variant, major, minor, patch);
    }

    std::string VkResult_to_string(VkResult result)
    {
        if (VkResult_strmap.contains(result))
        {
            return VkResult_strmap[result];
        }
        return fmt::format(
            "undocumented VkResult: {}",
            (int)result
        );
    }

    uint32_t VulkanApiVersion_encode(VulkanApiVersion version)
    {
        switch (version)
        {
        case VulkanApiVersion::Vulkan1_0:
            return VK_API_VERSION_1_0;
        case VulkanApiVersion::Vulkan1_1:
            return VK_API_VERSION_1_1;
        case VulkanApiVersion::Vulkan1_2:
            return VK_API_VERSION_1_2;
        case VulkanApiVersion::Vulkan1_3:
            return VK_API_VERSION_1_3;
        default:
            return VK_API_VERSION_1_0;
        }
    }

    ExtensionProperties ExtensionProperties_from_vk(
        VkExtensionProperties vk_properties
    )
    {
        return ExtensionProperties{
            .name = cstr_to_std(vk_properties.extensionName),
            .spec_version = vk_properties.specVersion
        };
    }

    LayerProperties LayerProperties_from_vk(
        const VkLayerProperties& vk_properties
    )
    {
        return LayerProperties{
            .name = cstr_to_std(vk_properties.layerName),
            .spec_version = Version(vk_properties.specVersion),
            .implementation_version = vk_properties.implementationVersion,
            .description = cstr_to_std(vk_properties.description)
        };
    }

    PhysicalDeviceLimits PhysicalDeviceLimits_from_vk(
        const VkPhysicalDeviceLimits& vk_limits
    )
    {
        return PhysicalDeviceLimits{
            .max_image_dimension_1d =
            vk_limits.maxImageDimension1D,

            .max_image_dimension_2d =
            vk_limits.maxImageDimension2D,

            .max_image_dimension_3d =
            vk_limits.maxImageDimension3D,

            .max_image_dimension_cube =
            vk_limits.maxImageDimensionCube,

            .max_image_array_layers =
            vk_limits.maxImageArrayLayers,

            .max_texel_buffer_elements =
            vk_limits.maxTexelBufferElements,

            .max_uniform_buffer_range =
            vk_limits.maxUniformBufferRange,

            .max_storage_buffer_range =
            vk_limits.maxStorageBufferRange,

            .max_push_constants_size =
            vk_limits.maxPushConstantsSize,

            .max_memory_allocation_count =
            vk_limits.maxMemoryAllocationCount,

            .max_sampler_allocation_count =
            vk_limits.maxSamplerAllocationCount,

            .buffer_image_granularity =
            vk_limits.bufferImageGranularity,

            .sparse_address_space_size =
            vk_limits.sparseAddressSpaceSize,

            .max_bound_descriptor_sets =
            vk_limits.maxBoundDescriptorSets,

            .max_per_stage_descriptor_samplers =
            vk_limits.maxPerStageDescriptorSamplers,

            .max_per_stage_descriptor_uniform_buffers =
            vk_limits.maxPerStageDescriptorUniformBuffers,

            .max_per_stage_descriptor_storage_buffers =
            vk_limits.maxPerStageDescriptorStorageBuffers,

            .max_per_stage_descriptor_sampled_images =
            vk_limits.maxPerStageDescriptorSampledImages,

            .max_per_stage_descriptor_storage_images =
            vk_limits.maxPerStageDescriptorStorageImages,

            .max_per_stage_descriptor_input_attachments =
            vk_limits.maxPerStageDescriptorInputAttachments,

            .max_per_stage_resources =
            vk_limits.maxPerStageResources,

            .max_descriptor_set_samplers =
            vk_limits.maxDescriptorSetSamplers,

            .max_descriptor_set_uniform_buffers =
            vk_limits.maxDescriptorSetUniformBuffers,

            .max_descriptor_set_uniform_buffers_dynamic =
            vk_limits.maxDescriptorSetUniformBuffersDynamic,

            .max_descriptor_set_storage_buffers =
            vk_limits.maxDescriptorSetStorageBuffers,

            .max_descriptor_set_storage_buffers_dynamic =
            vk_limits.maxDescriptorSetStorageBuffersDynamic,

            .max_descriptor_set_sampled_images =
            vk_limits.maxDescriptorSetSampledImages,

            .max_descriptor_set_storage_images =
            vk_limits.maxDescriptorSetStorageImages,

            .max_descriptor_set_input_attachments =
            vk_limits.maxDescriptorSetInputAttachments,

            .max_vertex_input_attributes =
            vk_limits.maxVertexInputAttributes,

            .max_vertex_input_bindings =
            vk_limits.maxVertexInputBindings,

            .max_vertex_input_attribute_offset =
            vk_limits.maxVertexInputAttributeOffset,

            .max_vertex_input_binding_stride =
            vk_limits.maxVertexInputBindingStride,

            .max_vertex_output_components =
            vk_limits.maxVertexOutputComponents,

            .max_tessellation_generation_level =
            vk_limits.maxTessellationGenerationLevel,

            .max_tessellation_patch_size =
            vk_limits.maxTessellationPatchSize,

            .max_tessellation_control_per_vertex_input_components =
            vk_limits
            .maxTessellationControlPerVertexInputComponents,

            .max_tessellation_control_per_vertex_output_components =
            vk_limits
            .maxTessellationControlPerVertexOutputComponents,

            .max_tessellation_control_per_patch_output_components =
            vk_limits
            .maxTessellationControlPerPatchOutputComponents,

            .max_tessellation_control_total_output_components =
            vk_limits
            .maxTessellationControlTotalOutputComponents,

            .max_tessellation_evaluation_input_components =
            vk_limits.maxTessellationEvaluationInputComponents,

            .max_tessellation_evaluation_output_components =
            vk_limits.maxTessellationEvaluationOutputComponents,

            .max_geometry_shader_invocations =
            vk_limits.maxGeometryShaderInvocations,

            .max_geometry_input_components =
            vk_limits.maxGeometryInputComponents,

            .max_geometry_output_components =
            vk_limits.maxGeometryOutputComponents,

            .max_geometry_output_vertices =
            vk_limits.maxGeometryOutputVertices,

            .max_geometry_total_output_components =
            vk_limits.maxGeometryTotalOutputComponents,

            .max_fragment_input_components =
            vk_limits.maxFragmentInputComponents,

            .max_fragment_output_attachments =
            vk_limits.maxFragmentOutputAttachments,

            .max_fragment_dual_src_attachments =
            vk_limits.maxFragmentDualSrcAttachments,

            .max_fragment_combined_output_resources =
            vk_limits.maxFragmentCombinedOutputResources,

            .max_compute_shared_memory_size =
            vk_limits.maxComputeSharedMemorySize,

            .max_compute_work_group_count = raw_arr_to_std<3>(
                vk_limits.maxComputeWorkGroupCount
            ),

            .max_compute_work_group_invocations =
            vk_limits.maxComputeWorkGroupInvocations,

            .max_compute_work_group_size = raw_arr_to_std<3>(
                vk_limits.maxComputeWorkGroupSize
            ),

            .sub_pixel_precision_bits =
            vk_limits.subPixelPrecisionBits,

            .sub_texel_precision_bits =
            vk_limits.subTexelPrecisionBits,

            .mipmap_precision_bits =
            vk_limits.mipmapPrecisionBits,

            .max_draw_indexed_index_value =
            vk_limits.maxDrawIndexedIndexValue,

            .max_draw_indirect_count =
            vk_limits.maxDrawIndirectCount,

            .max_sampler_lod_bias = vk_limits.maxSamplerLodBias,

            .max_sampler_anisotropy =
            vk_limits.maxSamplerAnisotropy,

            .max_viewports = vk_limits.maxViewports,

            .max_viewport_dimensions = raw_arr_to_std<2>(
                vk_limits.maxViewportDimensions
            ),

            .viewport_bounds_range = raw_arr_to_std<2>(
                vk_limits.viewportBoundsRange
            ),

            .viewport_sub_pixel_bits =
            vk_limits.viewportSubPixelBits,

            .min_memory_map_alignment =
            vk_limits.minMemoryMapAlignment,

            .min_texel_buffer_offset_alignment =
            vk_limits.minTexelBufferOffsetAlignment,

            .min_uniform_buffer_offset_alignment =
            vk_limits.minUniformBufferOffsetAlignment,

            .min_storage_buffer_offset_alignment =
            vk_limits.minStorageBufferOffsetAlignment,

            .min_texel_offset = vk_limits.minTexelOffset,

            .max_texel_offset = vk_limits.maxTexelOffset,

            .min_texel_gather_offset =
            vk_limits.minTexelGatherOffset,

            .max_texel_gather_offset =
            vk_limits.maxTexelGatherOffset,

            .min_interpolation_offset =
            vk_limits.minInterpolationOffset,

            .max_interpolation_offset =
            vk_limits.maxInterpolationOffset,

            .sub_pixel_interpolation_offset_bits =
            vk_limits.subPixelInterpolationOffsetBits,

            .max_framebuffer_width =
            vk_limits.maxFramebufferWidth,

            .max_framebuffer_height =
            vk_limits.maxFramebufferHeight,

            .max_framebuffer_layers =
            vk_limits.maxFramebufferLayers,

            .framebuffer_color_sample_counts =
            vk_limits.framebufferColorSampleCounts,

            .framebuffer_depth_sample_counts =
            vk_limits.framebufferDepthSampleCounts,

            .framebuffer_stencil_sample_counts =
            vk_limits.framebufferStencilSampleCounts,

            .framebuffer_no_attachments_sample_counts =
            vk_limits.framebufferNoAttachmentsSampleCounts,

            .max_color_attachments =
            vk_limits.maxColorAttachments,

            .sampled_image_color_sample_counts =
            vk_limits.sampledImageColorSampleCounts,

            .sampled_image_integer_sample_counts =
            vk_limits.sampledImageIntegerSampleCounts,

            .sampled_image_depth_sample_counts =
            vk_limits.sampledImageDepthSampleCounts,

            .sampled_image_stencil_sample_counts =
            vk_limits.sampledImageStencilSampleCounts,

            .storage_image_sample_counts =
            vk_limits.storageImageSampleCounts,

            .max_sample_mask_words =
            vk_limits.maxSampleMaskWords,

            .timestamp_compute_and_graphics =
            (bool)vk_limits.timestampComputeAndGraphics,

            .timestamp_period = vk_limits.timestampPeriod,

            .max_clip_distances = vk_limits.maxClipDistances,

            .max_cull_distances = vk_limits.maxCullDistances,

            .max_combined_clip_and_cull_distances =
            vk_limits.maxCombinedClipAndCullDistances,

            .discrete_queue_priorities =
            vk_limits.discreteQueuePriorities,

            .point_size_range = raw_arr_to_std<2>(
                vk_limits.pointSizeRange
            ),

            .line_width_range = raw_arr_to_std<2>(
                vk_limits.lineWidthRange
            ),

            .point_size_granularity =
            vk_limits.pointSizeGranularity,

            .line_width_granularity =
            vk_limits.lineWidthGranularity,

            .strict_lines = (bool)vk_limits.strictLines,

            .standard_sample_locations =
            (bool)vk_limits.standardSampleLocations,

            .optimal_buffer_copy_offset_alignment =
            vk_limits.optimalBufferCopyOffsetAlignment,

            .optimal_buffer_copy_row_pitch_alignment =
            vk_limits.optimalBufferCopyRowPitchAlignment,

            .non_coherent_atom_size =
            vk_limits.nonCoherentAtomSize
        };
    }

    PhysicalDeviceSparseProperties PhysicalDeviceSparseProperties_from_vk(
        const VkPhysicalDeviceSparseProperties& vk_properties
    )
    {
        return PhysicalDeviceSparseProperties{
            .residency_standard_2d_block_shape =
            (bool)vk_properties.residencyStandard2DBlockShape,

            .residency_standard_2d_multisample_block_shape =
            (bool)vk_properties.residencyStandard2DMultisampleBlockShape,

            .residency_standard_3d_block_shape =
            (bool)vk_properties.residencyStandard3DBlockShape,

            .residency_aligned_mip_size =
            (bool)vk_properties.residencyAlignedMipSize,

            .residency_non_resident_strict =
            (bool)vk_properties.residencyNonResidentStrict
        };
    }

    PhysicalDeviceProperties PhysicalDeviceProperties_from_vk(
        const VkPhysicalDeviceProperties& vk_properties
    )
    {
        return PhysicalDeviceProperties{
            .api_version = Version(vk_properties.apiVersion),
            .driver_version = vk_properties.driverVersion,
            .vendor_id = vk_properties.vendorID,
            .device_id = vk_properties.deviceID,
            .device_type = vk_properties.deviceType,
            .device_name = cstr_to_std(vk_properties.deviceName),
            .pipeline_cache_uuid = raw_arr_to_std<VK_UUID_SIZE>(
                vk_properties.pipelineCacheUUID
            ),
            .limits = PhysicalDeviceLimits_from_vk(vk_properties.limits),
            .sparse_properties = PhysicalDeviceSparseProperties_from_vk(
                vk_properties.sparseProperties
            )
        };
    }

    PhysicalDeviceFeatures PhysicalDeviceFeatures_from_vk(
        const VkPhysicalDeviceFeatures& vk_features
    )
    {
        return PhysicalDeviceFeatures{
            .robust_buffer_access =
            (bool)vk_features.robustBufferAccess,

            .full_draw_index_uint32 =
            (bool)vk_features.fullDrawIndexUint32,

            .image_cube_array =
            (bool)vk_features.imageCubeArray,

            .independent_blend =
            (bool)vk_features.independentBlend,

            .geometry_shader =
            (bool)vk_features.geometryShader,

            .tessellation_shader =
            (bool)vk_features.tessellationShader,

            .sample_rate_shading =
            (bool)vk_features.sampleRateShading,

            .dual_src_blend =
            (bool)vk_features.dualSrcBlend,

            .logic_op =
            (bool)vk_features.logicOp,

            .multi_draw_indirect =
            (bool)vk_features.multiDrawIndirect,

            .draw_indirect_first_instance =
            (bool)vk_features.drawIndirectFirstInstance,

            .depth_clamp =
            (bool)vk_features.depthClamp,

            .depth_bias_clamp =
            (bool)vk_features.depthBiasClamp,

            .fill_mode_non_solid =
            (bool)vk_features.fillModeNonSolid,

            .depth_bounds =
            (bool)vk_features.depthBounds,

            .wide_lines =
            (bool)vk_features.wideLines,

            .large_points =
            (bool)vk_features.largePoints,

            .alpha_to_one =
            (bool)vk_features.alphaToOne,

            .multi_viewport =
            (bool)vk_features.multiViewport,

            .sampler_anisotropy =
            (bool)vk_features.samplerAnisotropy,

            .texture_compression_etc2 =
            (bool)vk_features.textureCompressionETC2,

            .texture_compression_astc_ldr =
            (bool)vk_features.textureCompressionASTC_LDR,

            .texture_compression_bc =
            (bool)vk_features.textureCompressionBC,

            .occlusion_query_precise =
            (bool)vk_features.occlusionQueryPrecise,

            .pipeline_statistics_query =
            (bool)vk_features.pipelineStatisticsQuery,

            .vertex_pipeline_stores_and_atomics =
            (bool)vk_features.vertexPipelineStoresAndAtomics,

            .fragment_stores_and_atomics =
            (bool)vk_features.fragmentStoresAndAtomics,

            .shader_tessellation_and_geometry_point_size =
            (bool)vk_features
            .shaderTessellationAndGeometryPointSize,

            .shader_image_gather_extended =
            (bool)vk_features.shaderImageGatherExtended,

            .shader_storage_image_extended_formats =
            (bool)vk_features
            .shaderStorageImageExtendedFormats,

            .shader_storage_image_multisample =
            (bool)vk_features.shaderStorageImageMultisample,

            .shader_storage_image_read_without_format =
            (bool)vk_features
            .shaderStorageImageReadWithoutFormat,

            .shader_storage_image_write_without_format =
            (bool)vk_features
            .shaderStorageImageWriteWithoutFormat,

            .shader_uniform_buffer_array_dynamic_indexing =
            (bool)vk_features
            .shaderUniformBufferArrayDynamicIndexing,

            .shader_sampled_image_array_dynamic_indexing =
            (bool)vk_features
            .shaderSampledImageArrayDynamicIndexing,

            .shader_storage_buffer_array_dynamic_indexing =
            (bool)vk_features
            .shaderStorageBufferArrayDynamicIndexing,

            .shader_storage_image_array_dynamic_indexing =
            (bool)vk_features
            .shaderStorageImageArrayDynamicIndexing,

            .shader_clip_distance =
            (bool)vk_features.shaderClipDistance,

            .shader_cull_distance =
            (bool)vk_features.shaderCullDistance,

            .shader_float64 =
            (bool)vk_features.shaderFloat64,

            .shader_int64 =
            (bool)vk_features.shaderInt64,

            .shader_int16 =
            (bool)vk_features.shaderInt16,

            .shader_resource_residency =
            (bool)vk_features.shaderResourceResidency,

            .shader_resource_min_lod =
            (bool)vk_features.shaderResourceMinLod,

            .sparse_binding =
            (bool)vk_features.sparseBinding,

            .sparse_residency_buffer =
            (bool)vk_features.sparseResidencyBuffer,

            .sparse_residency_image_2d =
            (bool)vk_features.sparseResidencyImage2D,

            .sparse_residency_image_3d =
            (bool)vk_features.sparseResidencyImage3D,

            .sparse_residency2_samples =
            (bool)vk_features.sparseResidency2Samples,

            .sparse_residency4_samples =
            (bool)vk_features.sparseResidency4Samples,

            .sparse_residency8_samples =
            (bool)vk_features.sparseResidency8Samples,

            .sparse_residency16_samples =
            (bool)vk_features.sparseResidency16Samples,

            .sparse_residency_aliased =
            (bool)vk_features.sparseResidencyAliased,

            .variable_multisample_rate =
            (bool)vk_features.variableMultisampleRate,

            .inherited_queries =
            (bool)vk_features.inheritedQueries
        };
    }

    VkPhysicalDeviceFeatures PhysicalDeviceFeatures_to_vk(
        const PhysicalDeviceFeatures& features
    )
    {
        return VkPhysicalDeviceFeatures{
            .robustBufferAccess =
            features.robust_buffer_access,

            .fullDrawIndexUint32 =
            features.full_draw_index_uint32,

            .imageCubeArray =
            features.image_cube_array,

            .independentBlend =
            features.independent_blend,

            .geometryShader =
            features.geometry_shader,

            .tessellationShader =
            features.tessellation_shader,

            .sampleRateShading =
            features.sample_rate_shading,

            .dualSrcBlend =
            features.dual_src_blend,

            .logicOp =
            features.logic_op,

            .multiDrawIndirect =
            features.multi_draw_indirect,

            .drawIndirectFirstInstance =
            features.draw_indirect_first_instance,

            .depthClamp =
            features.depth_clamp,

            .depthBiasClamp =
            features.depth_bias_clamp,

            .fillModeNonSolid =
            features.fill_mode_non_solid,

            .depthBounds =
            features.depth_bounds,

            .wideLines =
            features.wide_lines,

            .largePoints =
            features.large_points,

            .alphaToOne =
            features.alpha_to_one,

            .multiViewport =
            features.multi_viewport,

            .samplerAnisotropy =
            features.sampler_anisotropy,

            .textureCompressionETC2 =
            features.texture_compression_etc2,

            .textureCompressionASTC_LDR =
            features.texture_compression_astc_ldr,

            .textureCompressionBC =
            features.texture_compression_bc,

            .occlusionQueryPrecise =
            features.occlusion_query_precise,

            .pipelineStatisticsQuery =
            features.pipeline_statistics_query,

            .vertexPipelineStoresAndAtomics =
            features.vertex_pipeline_stores_and_atomics,

            .fragmentStoresAndAtomics =
            features.fragment_stores_and_atomics,

            .shaderTessellationAndGeometryPointSize = features
            .shader_tessellation_and_geometry_point_size,

            .shaderImageGatherExtended =
            features.shader_image_gather_extended,

            .shaderStorageImageExtendedFormats =
            features.shader_storage_image_extended_formats,

            .shaderStorageImageMultisample =
            features.shader_storage_image_multisample,

            .shaderStorageImageReadWithoutFormat =
            features.shader_storage_image_read_without_format,

            .shaderStorageImageWriteWithoutFormat =
            features.shader_storage_image_write_without_format,

            .shaderUniformBufferArrayDynamicIndexing = features
            .shader_uniform_buffer_array_dynamic_indexing,

            .shaderSampledImageArrayDynamicIndexing = features
            .shader_sampled_image_array_dynamic_indexing,

            .shaderStorageBufferArrayDynamicIndexing = features
            .shader_storage_buffer_array_dynamic_indexing,

            .shaderStorageImageArrayDynamicIndexing = features
            .shader_storage_image_array_dynamic_indexing,

            .shaderClipDistance =
            features.shader_clip_distance,

            .shaderCullDistance =
            features.shader_cull_distance,

            .shaderFloat64 =
            features.shader_float64,

            .shaderInt64 =
            features.shader_int64,

            .shaderInt16 =
            features.shader_int16,

            .shaderResourceResidency =
            features.shader_resource_residency,

            .shaderResourceMinLod =
            features.shader_resource_min_lod,

            .sparseBinding =
            features.sparse_binding,

            .sparseResidencyBuffer =
            features.sparse_residency_buffer,

            .sparseResidencyImage2D =
            features.sparse_residency_image_2d,

            .sparseResidencyImage3D =
            features.sparse_residency_image_3d,

            .sparseResidency2Samples =
            features.sparse_residency2_samples,

            .sparseResidency4Samples =
            features.sparse_residency4_samples,

            .sparseResidency8Samples =
            features.sparse_residency8_samples,

            .sparseResidency16Samples =
            features.sparse_residency16_samples,

            .sparseResidencyAliased =
            features.sparse_residency_aliased,

            .variableMultisampleRate =
            features.variable_multisample_rate,

            .inheritedQueries =
            features.inherited_queries,
        };
    }

    Extent3d Extent3d_from_vk(const VkExtent3D& vk_extent_3d)
    {
        return Extent3d{
            .width = vk_extent_3d.width,
            .height = vk_extent_3d.height,
            .depth = vk_extent_3d.depth
        };
    }

    VkExtent3D Extent3d_to_vk(const Extent3d& extent_3d)
    {
        return VkExtent3D{
            .width = extent_3d.width,
            .height = extent_3d.height,
            .depth = extent_3d.depth
        };
    }

    Extent2d Extent2d_from_vk(const VkExtent2D& vk_extent_2d)
    {
        return Extent2d{
            .width = vk_extent_2d.width,
            .height = vk_extent_2d.height
        };
    }

    VkExtent2D Extent2d_to_vk(const Extent2d& extent_2d)
    {
        return VkExtent2D{
            .width = extent_2d.width,
            .height = extent_2d.height
        };
    }

    QueueFamily QueueFamily_from_vk(const VkQueueFamilyProperties& vk_family)
    {
        return QueueFamily{
            .queue_flags = vk_family.queueFlags,
            .queue_count = vk_family.queueCount,
            .timestamp_valid_bits = vk_family.timestampValidBits,
            .min_image_transfer_granularity = Extent3d_from_vk(
                vk_family.minImageTransferGranularity
            )
        };
    }

    SurfaceCapabilities SurfaceCapabilities_from_vk(
        const VkSurfaceCapabilitiesKHR& vk_capabilities
    )
    {
        return SurfaceCapabilities{
            .min_image_count = vk_capabilities.minImageCount,
            .max_image_count = vk_capabilities.maxImageCount,
            .current_extent = Extent2d_from_vk(vk_capabilities.currentExtent),
            .min_image_extent = Extent2d_from_vk(
                vk_capabilities.minImageExtent
            ),
            .max_image_extent = Extent2d_from_vk(
                vk_capabilities.maxImageExtent
            ),
            .max_image_array_layers = vk_capabilities.maxImageArrayLayers,
            .supported_transforms = vk_capabilities.supportedTransforms,
            .current_transform = vk_capabilities.currentTransform,

            .supported_composite_alpha =
            vk_capabilities.supportedCompositeAlpha,

            .supported_usage_flags = vk_capabilities.supportedUsageFlags
        };
    }

    SurfaceFormat SurfaceFormat_from_vk(
        const VkSurfaceFormatKHR& vk_surface_format
    )
    {
        return SurfaceFormat{
            .format = vk_surface_format.format,
            .color_space = vk_surface_format.colorSpace
        };
    }

    VkSurfaceFormatKHR SurfaceFormat_to_vk(const SurfaceFormat& surface_format)
    {
        return VkSurfaceFormatKHR{
            .format = surface_format.format,
            .colorSpace = surface_format.color_space
        };
    }

    DebugLabel DebugLabel_from_vk(const VkDebugUtilsLabelEXT& vk_label)
    {
        return DebugLabel{
            .name = cstr_to_std(vk_label.pLabelName),
            .color = raw_arr_to_std<4>(vk_label.color)
        };
    }

    DebugObjectInfo DebugObjectInfo_from_vk(
        const VkDebugUtilsObjectNameInfoEXT& vk_info
    )
    {
        return DebugObjectInfo{
            .type = vk_info.objectType,
            .handle = vk_info.objectHandle,
            .name = cstr_to_std(vk_info.pObjectName)
        };
    }

    DebugMessageData DebugMessageData_from_vk(
        const VkDebugUtilsMessengerCallbackDataEXT& vk_data
    )
    {
        std::vector<DebugLabel> queue_labels;
        for (size_t i = 0; i < vk_data.queueLabelCount; i++)
        {
            queue_labels.push_back(
                DebugLabel_from_vk(vk_data.pQueueLabels[i])
            );
        }

        std::vector<DebugLabel> cmd_buf_labels;
        for (size_t i = 0; i < vk_data.cmdBufLabelCount; i++)
        {
            cmd_buf_labels.push_back(
                DebugLabel_from_vk(vk_data.pCmdBufLabels[i])
            );
        }

        std::vector<DebugObjectInfo> objects;
        for (size_t i = 0; i < vk_data.objectCount; i++)
        {
            objects.push_back(
                DebugObjectInfo_from_vk(vk_data.pObjects[i])
            );
        }

        return DebugMessageData{
            .message_id_name = cstr_to_std(vk_data.pMessageIdName),
            .message_id_number = vk_data.messageIdNumber,
            .message = cstr_to_std(vk_data.pMessage),
            .queue_labels = queue_labels,
            .cmd_buf_labels = cmd_buf_labels,
            .objects = objects
        };;
    }

    VkDeviceQueueCreateInfo QueueRequest_to_vk(
        const QueueRequest& request,
        std::vector<float>& waste_priorities
    )
    {
        waste_priorities = request.priorities;
        return VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = request.flags,
            .queueFamilyIndex = request.queue_family_index,
            .queueCount = request.num_queues_to_create,
            .pQueuePriorities = waste_priorities.data()
        };
    }

    VkComponentMapping ComponentMapping_to_vk(const ComponentMapping& mapping)
    {
        return VkComponentMapping{
            .r = mapping.r,
            .g = mapping.g,
            .b = mapping.b,
            .a = mapping.a
        };
    }

    VkImageSubresourceRange ImageSubresourceRange_to_vk(
        const ImageSubresourceRange& range
    )
    {
        return VkImageSubresourceRange{
            .aspectMask = range.aspect_mask,
            .baseMipLevel = range.base_mip_level,
            .levelCount = range.level_count,
            .baseArrayLayer = range.base_array_layer,
            .layerCount = range.layer_count
        };
    }

    VkSpecializationMapEntry SpecializationMapEntry_to_vk(
        const SpecializationMapEntry& entry
    )
    {
        return VkSpecializationMapEntry{
            .constantID = entry.constant_id,
            .offset = entry.offset,
            .size = entry.size
        };
    }

    VkSpecializationInfo SpecializationInfo_to_vk(
        const SpecializationInfo& info,
        std::vector<VkSpecializationMapEntry>& waste_vk_map_entries,
        std::vector<uint8_t>& waste_data
    )
    {
        waste_vk_map_entries.resize(info.map_entries.size());
        for (size_t i = 0; i < info.map_entries.size(); i++)
        {
            waste_vk_map_entries[i] = SpecializationMapEntry_to_vk(
                info.map_entries[i]
            );
        }

        waste_data = info.data;

        return VkSpecializationInfo{
            .mapEntryCount = (uint32_t)waste_vk_map_entries.size(),
            .pMapEntries = waste_vk_map_entries.data(),
            .dataSize = waste_data.size(),
            .pData = reinterpret_cast<const void*>(waste_data.data())
        };
    }

    VkPipelineShaderStageCreateInfo ShaderStage_to_vk(
        const ShaderStage& stage,
        VkSpecializationInfo& waste_vk_specialization_info,
        std::vector<VkSpecializationMapEntry>& waste_vk_map_entries,
        std::vector<uint8_t>& waste_data
    )
    {
        if (stage.specialization_info.has_value())
        {
            waste_vk_specialization_info = SpecializationInfo_to_vk(
                stage.specialization_info.value(),
                waste_vk_map_entries,
                waste_data
            );
        }

        return VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = stage.flags,
            .stage = stage.stage,
            .module = lock_wptr(stage.module)->handle(),
            .pName = stage.entry_point.c_str(),

            .pSpecializationInfo = stage.specialization_info.has_value()
            ? &waste_vk_specialization_info
            : nullptr
        };
    }

    VkPipelineDynamicStateCreateInfo DynamicStates_to_vk(
        const DynamicStates& states,
        std::vector<VkDynamicState>& waste_dynamic_states
    )
    {
        waste_dynamic_states = states;
        return VkPipelineDynamicStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .dynamicStateCount = (uint32_t)waste_dynamic_states.size(),
            .pDynamicStates = waste_dynamic_states.data()
        };
    }

    VkVertexInputBindingDescription VertexInputBindingDescription_to_vk(
        const VertexInputBindingDescription& description
    )
    {
        return VkVertexInputBindingDescription{
            .binding = description.binding,
            .stride = description.stride,
            .inputRate = description.input_rate
        };
    }

    VkVertexInputAttributeDescription VertexInputAttributeDescription_to_vk(
        const VertexInputAttributeDescription& description
    )
    {
        return VkVertexInputAttributeDescription{
            .location = description.location,
            .binding = description.binding,
            .format = description.format,
            .offset = description.offset
        };
    }

    VkPipelineVertexInputStateCreateInfo VertexInputState_to_vk(
        const VertexInputState& state,

        std::vector<VkVertexInputBindingDescription>&
        waste_vk_binding_descriptions,

        std::vector<VkVertexInputAttributeDescription>&
        waste_vk_attribute_descriptions
    )
    {
        waste_vk_binding_descriptions.resize(state.binding_descriptions.size());
        for (size_t i = 0; i < state.binding_descriptions.size(); i++)
        {
            waste_vk_binding_descriptions[i] =
                VertexInputBindingDescription_to_vk(
                    state.binding_descriptions[i]
                );
        }

        waste_vk_attribute_descriptions.resize(
            state.attribute_descriptions.size()
        );
        for (size_t i = 0; i < state.attribute_descriptions.size(); i++)
        {
            waste_vk_attribute_descriptions[i] =
                VertexInputAttributeDescription_to_vk(
                    state.attribute_descriptions[i]
                );
        }

        return VkPipelineVertexInputStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,

            .vertexBindingDescriptionCount =
            (uint32_t)waste_vk_binding_descriptions.size(),

            .pVertexBindingDescriptions = waste_vk_binding_descriptions.data(),

            .vertexAttributeDescriptionCount =
            (uint32_t)waste_vk_attribute_descriptions.size(),

            .pVertexAttributeDescriptions =
            waste_vk_attribute_descriptions.data()
        };
    }

    VkPipelineInputAssemblyStateCreateInfo InputAssemblyState_to_vk(
        const InputAssemblyState& state
    )
    {
        return VkPipelineInputAssemblyStateCreateInfo{
            .sType =
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,

            .pNext = nullptr,
            .flags = 0,
            .topology = state.topology,
            .primitiveRestartEnable = state.primitive_restart_enable
        };
    }

    VkPipelineTessellationStateCreateInfo TessellationState_to_vk(
        const TessellationState& state
    )
    {
        return VkPipelineTessellationStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .patchControlPoints = state.patch_control_points
        };
    }

    VkViewport Viewport_to_vk(const Viewport& viewport)
    {
        return VkViewport{
            .x = viewport.x,
            .y = viewport.y,
            .width = viewport.width,
            .height = viewport.height,
            .minDepth = viewport.min_depth,
            .maxDepth = viewport.max_depth
        };
    }

    VkOffset2D Offset2d_to_vk(const Offset2d& offset)
    {
        return VkOffset2D{
            .x = offset.x,
            .y = offset.y
        };
    }

    VkOffset3D Offset3d_to_vk(const Offset3d& offset)
    {
        return VkOffset3D{
            .x = offset.x,
            .y = offset.y,
            .z = offset.z
        };
    }

    VkRect2D Rect2d_to_vk(const Rect2d& rect)
    {
        return VkRect2D{
            .offset = Offset2d_to_vk(rect.offset),
            .extent = Extent2d_to_vk(rect.extent)
        };
    }

    VkPipelineViewportStateCreateInfo ViewportState_to_vk(
        const ViewportState& state,
        std::vector<VkViewport>& waste_vk_viewports,
        std::vector<VkRect2D>& waste_vk_scissors
    )
    {
        waste_vk_viewports.resize(state.viewports.size());
        for (size_t i = 0; i < state.viewports.size(); i++)
        {
            waste_vk_viewports[i] = Viewport_to_vk(state.viewports[i]);
        }

        waste_vk_scissors.resize(state.scissors.size());
        for (size_t i = 0; i < state.scissors.size(); i++)
        {
            waste_vk_scissors[i] = Rect2d_to_vk(state.scissors[i]);
        }

        return VkPipelineViewportStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .viewportCount = (uint32_t)waste_vk_viewports.size(),
            .pViewports = waste_vk_viewports.data(),
            .scissorCount = (uint32_t)waste_vk_scissors.size(),
            .pScissors = waste_vk_scissors.data()
        };
    }

    VkPipelineRasterizationStateCreateInfo RasterizationState_to_vk(
        const RasterizationState& state
    )
    {
        return VkPipelineRasterizationStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .depthClampEnable = state.depth_clamp_enable,
            .rasterizerDiscardEnable = state.rasterizer_discard_enable,
            .polygonMode = state.polygon_mode,
            .cullMode = state.cull_mode,
            .frontFace = state.front_face,
            .depthBiasEnable = state.depth_bias_enable,
            .depthBiasConstantFactor = state.depth_bias_constant_factor,
            .depthBiasClamp = state.depth_bias_clamp,
            .depthBiasSlopeFactor = state.depth_bias_slope_factor,
            .lineWidth = state.line_width
        };
    }

    VkPipelineMultisampleStateCreateInfo MultisampleState_to_vk(
        const MultisampleState& state,
        std::vector<VkSampleMask>& waste_sample_mask
    )
    {
        waste_sample_mask = state.sample_mask;
        return VkPipelineMultisampleStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .rasterizationSamples = state.rasterization_samples,
            .sampleShadingEnable = state.sample_shading_enable,
            .minSampleShading = state.min_sample_shading,
            .pSampleMask = waste_sample_mask.data(),
            .alphaToCoverageEnable = state.alpha_to_coverage_enable,
            .alphaToOneEnable = state.alpha_to_one_enable
        };
    }

    VkStencilOpState StencilOpState_to_vk(const StencilOpState& state)
    {
        return VkStencilOpState{
            .failOp = state.fail_op,
            .passOp = state.pass_op,
            .depthFailOp = state.depth_fail_op,
            .compareOp = state.compare_op,
            .compareMask = state.compare_mask,
            .writeMask = state.write_mask,
            .reference = state.reference
        };
    }

    VkPipelineDepthStencilStateCreateInfo DepthStencilState_to_vk(
        const DepthStencilState& state
    )
    {
        return VkPipelineDepthStencilStateCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = state.flags,
            .depthTestEnable = state.depth_test_enable,
            .depthWriteEnable = state.depth_write_enable,
            .depthCompareOp = state.depth_compare_op,
            .depthBoundsTestEnable = state.depth_bounds_test_enable,
            .stencilTestEnable = state.stencil_test_enable,
            .front = StencilOpState_to_vk(state.front),
            .back = StencilOpState_to_vk(state.back),
            .minDepthBounds = state.min_depth_bounds,
            .maxDepthBounds = state.max_depth_bounds
        };
    }

    VkPipelineColorBlendAttachmentState ColorBlendAttachment_to_vk(
        const ColorBlendAttachment& attachment
    )
    {
        return VkPipelineColorBlendAttachmentState{
            .blendEnable = attachment.blend_enable,
            .srcColorBlendFactor = attachment.src_color_blend_factor,
            .dstColorBlendFactor = attachment.dst_color_blend_factor,
            .colorBlendOp = attachment.color_blend_op,
            .srcAlphaBlendFactor = attachment.src_alpha_blend_factor,
            .dstAlphaBlendFactor = attachment.dst_alpha_blend_factor,
            .alphaBlendOp = attachment.alpha_blend_op,
            .colorWriteMask = attachment.color_write_mask
        };
    }

    VkPipelineColorBlendStateCreateInfo ColorBlendState_to_vk(
        const ColorBlendState& state,

        std::vector<VkPipelineColorBlendAttachmentState>&
        waste_vk_color_blend_attachments
    )
    {
        waste_vk_color_blend_attachments.resize(state.attachments.size());
        for (size_t i = 0; i < state.attachments.size(); i++)
        {
            waste_vk_color_blend_attachments[i] = ColorBlendAttachment_to_vk(
                state.attachments[i]
            );
        }

        VkPipelineColorBlendStateCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = state.flags,
            .logicOpEnable = state.logic_op_enable,
            .logicOp = state.logic_op,

            .attachmentCount =
            (uint32_t)waste_vk_color_blend_attachments.size(),

            .pAttachments = waste_vk_color_blend_attachments.data()
        };
        std::copy(
            state.blend_constants.data(),
            state.blend_constants.data() + 4,
            info.blendConstants
        );
        return info;
    }

    VkDescriptorSetLayoutBinding DescriptorSetLayoutBinding_to_vk(
        const DescriptorSetLayoutBinding& binding,
        std::vector<VkSampler>& waste_vk_immutable_samplers
    )
    {
        waste_vk_immutable_samplers.resize(binding.immutable_samplers.size());
        for (size_t i = 0; i < binding.immutable_samplers.size(); i++)
        {
            waste_vk_immutable_samplers[i] =
                lock_wptr(binding.immutable_samplers[i])->handle();
        }

        return VkDescriptorSetLayoutBinding{
            .binding = binding.binding,
            .descriptorType = binding.descriptor_type,
            .descriptorCount = binding.descriptor_count,
            .stageFlags = binding.stage_flags,

            .pImmutableSamplers =
            waste_vk_immutable_samplers.empty()
            ? nullptr
            : waste_vk_immutable_samplers.data()
        };
    }

    VkPushConstantRange PushConstantRange_to_vk(const PushConstantRange& range)
    {
        return VkPushConstantRange{
            .stageFlags = range.stage_flags,
            .offset = range.offset,
            .size = range.size
        };
    }

    VkAttachmentDescription Attachment_to_vk(
        const Attachment& attachment
    )
    {
        return VkAttachmentDescription{
            .flags = attachment.flags,
            .format = attachment.format,
            .samples = attachment.samples,
            .loadOp = attachment.load_op,
            .storeOp = attachment.store_op,
            .stencilLoadOp = attachment.stencil_load_op,
            .stencilStoreOp = attachment.stencil_store_op,
            .initialLayout = attachment.initial_layout,
            .finalLayout = attachment.final_layout
        };
    }

    VkAttachmentReference AttachmentReference_to_vk(
        const AttachmentReference& ref
    )
    {
        return VkAttachmentReference{
            .attachment = ref.attachment,
            .layout = ref.layout
        };
    }

    VkSubpassDescription Subpass_to_vk(
        const Subpass& subpass,
        std::vector<VkAttachmentReference>& waste_vk_input_attachments,
        std::vector<VkAttachmentReference>& waste_vk_color_attachments,
        std::vector<VkAttachmentReference>& waste_vk_resolve_attachments,
        VkAttachmentReference& waste_vk_depth_stencil_attachment,
        std::vector<uint32_t>& waste_preserve_attachment_indices
    )
    {
        waste_vk_input_attachments.resize(subpass.input_attachments.size());
        for (size_t i = 0; i < subpass.input_attachments.size(); i++)
        {
            waste_vk_input_attachments[i] = AttachmentReference_to_vk(
                subpass.input_attachments[i]
            );
        }

        waste_vk_color_attachments.resize(subpass.color_attachments.size());
        for (size_t i = 0; i < subpass.color_attachments.size(); i++)
        {
            waste_vk_color_attachments[i] = AttachmentReference_to_vk(
                subpass.color_attachments[i]
            );
        }

        waste_vk_resolve_attachments.resize(subpass.resolve_attachments.size());
        for (size_t i = 0; i < subpass.resolve_attachments.size(); i++)
        {
            waste_vk_resolve_attachments[i] = AttachmentReference_to_vk(
                subpass.resolve_attachments[i]
            );
        }

        if (subpass.depth_stencil_attachment.has_value())
        {
            waste_vk_depth_stencil_attachment = AttachmentReference_to_vk(
                subpass.depth_stencil_attachment.value()
            );
        }

        waste_preserve_attachment_indices = subpass.preserve_attachment_indices;

        return VkSubpassDescription{
            .flags = subpass.flags,
            .pipelineBindPoint = subpass.pipeline_bind_point,
            .inputAttachmentCount = (uint32_t)waste_vk_input_attachments.size(),
            .pInputAttachments = waste_vk_input_attachments.data(),
            .colorAttachmentCount = (uint32_t)waste_vk_color_attachments.size(),
            .pColorAttachments = waste_vk_color_attachments.data(),

            .pResolveAttachments =
            waste_vk_resolve_attachments.empty()
            ? nullptr
            : waste_vk_resolve_attachments.data(),

            .pDepthStencilAttachment =
            subpass.depth_stencil_attachment.has_value()
            ? &waste_vk_depth_stencil_attachment
            : nullptr,

            .preserveAttachmentCount =
            (uint32_t)waste_preserve_attachment_indices.size(),

            .pPreserveAttachments = waste_preserve_attachment_indices.data()
        };
    }

    VkSubpassDependency SubpassDependency_to_vk(const SubpassDependency& dep)
    {
        return VkSubpassDependency{
            .srcSubpass = dep.src_subpass,
            .dstSubpass = dep.dst_subpass,
            .srcStageMask = dep.src_stage_mask,
            .dstStageMask = dep.dst_stage_mask,
            .srcAccessMask = dep.src_access_mask,
            .dstAccessMask = dep.dst_access_mask,
            .dependencyFlags = dep.dependency_flags
        };
    }

    MemoryRequirements MemoryRequirements_from_vk(
        const VkMemoryRequirements& req
    )
    {
        return MemoryRequirements{
            .size = req.size,
            .alignment = req.alignment,
            .memory_type_bits = req.memoryTypeBits
        };
    }

    MemoryType MemoryType_from_vk(const VkMemoryType& type)
    {
        return MemoryType{
            .property_flags = type.propertyFlags,
            .heap_index = type.heapIndex
        };
    }

    MemoryHeap MemoryHeap_from_vk(const VkMemoryHeap& heap)
    {
        return MemoryHeap{
            .size = heap.size,
            .flags = heap.flags
        };
    }

    PhysicalDeviceMemoryProperties PhysicalDeviceMemoryProperties_from_vk(
        const VkPhysicalDeviceMemoryProperties& properties
    )
    {
        std::vector<MemoryType> memory_types(properties.memoryTypeCount);
        for (size_t i = 0; i < properties.memoryTypeCount; i++)
        {
            memory_types[i] = MemoryType_from_vk(properties.memoryTypes[i]);
        }

        std::vector<MemoryHeap> memory_heaps(properties.memoryHeapCount);
        for (size_t i = 0; i < properties.memoryHeapCount; i++)
        {
            memory_heaps[i] = MemoryHeap_from_vk(properties.memoryHeaps[i]);
        }

        return PhysicalDeviceMemoryProperties{
            .memory_types = memory_types,
            .memory_heaps = memory_heaps
        };
    }

    VkDescriptorPoolSize DescriptorPoolSize_to_vk(
        const DescriptorPoolSize& pool_size
    )
    {
        return VkDescriptorPoolSize{
            .type = pool_size.type,
            .descriptorCount = pool_size.descriptor_count
        };
    }

    VkDescriptorImageInfo DescriptorImageInfo_to_vk(
        const DescriptorImageInfo& info
    )
    {
        return VkDescriptorImageInfo{
            .sampler = info.sampler.has_value()
            ? lock_wptr(info.sampler.value())->handle()
            : nullptr,

            .imageView = info.image_view.has_value()
            ? lock_wptr(info.image_view.value())->handle()
            : nullptr,

            .imageLayout = info.image_layout
        };
    }

    VkDescriptorBufferInfo DescriptorBufferInfo_to_vk(
        const DescriptorBufferInfo& info
    )
    {
        return VkDescriptorBufferInfo{
            .buffer = lock_wptr(info.buffer)->handle(),
            .offset = info.offset,
            .range = info.range
        };
    }

    VkWriteDescriptorSet WriteDescriptorSet_to_vk(
        const WriteDescriptorSet& write,
        std::vector<VkDescriptorImageInfo>& waste_vk_image_infos,
        std::vector<VkDescriptorBufferInfo>& waste_vk_buffer_infos,
        std::vector<VkBufferView>& waste_vk_texel_buffer_views
    )
    {
        waste_vk_image_infos.resize(write.image_infos.size());
        for (size_t i = 0; i < write.image_infos.size(); i++)
        {
            waste_vk_image_infos[i] = DescriptorImageInfo_to_vk(
                write.image_infos[i]
            );
        }

        waste_vk_buffer_infos.resize(write.buffer_infos.size());
        for (size_t i = 0; i < write.buffer_infos.size(); i++)
        {
            waste_vk_buffer_infos[i] = DescriptorBufferInfo_to_vk(
                write.buffer_infos[i]
            );
        }

        waste_vk_texel_buffer_views.resize(write.texel_buffer_views.size());
        for (size_t i = 0; i < write.texel_buffer_views.size(); i++)
        {
            waste_vk_texel_buffer_views[i] =
                lock_wptr(write.texel_buffer_views[i])->handle();
        }

        return VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = lock_wptr(write.dst_set)->handle(),
            .dstBinding = write.dst_binding,
            .dstArrayElement = write.dst_array_element,
            .descriptorCount = write.descriptor_count,
            .descriptorType = write.descriptor_type,

            .pImageInfo = waste_vk_image_infos.empty()
            ? nullptr : waste_vk_image_infos.data(),

            .pBufferInfo = waste_vk_buffer_infos.empty()
            ? nullptr : waste_vk_buffer_infos.data(),

            .pTexelBufferView = waste_vk_texel_buffer_views.empty()
            ? nullptr : waste_vk_texel_buffer_views.data()
        };
    }

    VkCopyDescriptorSet CopyDescriptorSet_to_vk(const CopyDescriptorSet& copy)
    {
        return VkCopyDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
            .pNext = nullptr,
            .srcSet = lock_wptr(copy.src_set)->handle(),
            .srcBinding = copy.src_binding,
            .srcArrayElement = copy.src_array_element,
            .dstSet = lock_wptr(copy.dst_set)->handle(),
            .dstBinding = copy.dst_binding,
            .dstArrayElement = copy.dst_array_element,
            .descriptorCount = copy.descriptor_count
        };
    }

    FormatProperties FormatProperties_from_vk(
        const VkFormatProperties& properties
    )
    {
        return FormatProperties{
            .linear_tiling_features = properties.linearTilingFeatures,
            .optimal_tiling_features = properties.optimalTilingFeatures,
            .buffer_features = properties.bufferFeatures
        };
    }

    ImageFormatProperties ImageFormatProperties_from_vk(
        const VkImageFormatProperties& properties
    )
    {
        return ImageFormatProperties{
            .max_extent = Extent3d_from_vk(properties.maxExtent),
            .max_mip_levels = properties.maxMipLevels,
            .max_array_layers = properties.maxArrayLayers,
            .sample_counts = properties.sampleCounts,
            .max_resource_size = properties.maxResourceSize
        };
    }

#pragma endregion

#pragma region error handling

    Error::Error()
        : message("no error information provided"),
        _vk_result(std::nullopt),
        stringify_vk_result(false)
    {}

    Error::Error(std::string message)
        : message(std::move(message)),
        _vk_result(std::nullopt),
        stringify_vk_result(false)
    {}

    Error::Error(const VkResult vk_result)
        : message(""),
        _vk_result(vk_result),
        stringify_vk_result(true)
    {}

    Error::Error(
        std::string message,
        const std::optional<VkResult>& vk_result,
        bool vk_result_already_embedded_in_message
    )
        : message(std::move(message)),
        _vk_result(vk_result),
        stringify_vk_result(!vk_result_already_embedded_in_message)
    {}

    std::string Error::to_string() const
    {
        std::string s = message;
        if (vk_result().has_value() && stringify_vk_result)
        {
            if (!message.empty())
            {
                s += ": ";
            }
            s += VkResult_to_string(vk_result().value());
        }
        return s;
    }

#pragma endregion

#pragma region classes and object wrappers

    std::vector<ExtensionProperties> PhysicalDevice::fetch_available_extensions(
        const std::string& layer_name
    ) const
    {
        try
        {
            const char* layer_name_cstr = nullptr;
            if (!layer_name.empty())
            {
                layer_name_cstr = layer_name.c_str();
            }

            uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(
                handle(),
                layer_name_cstr,
                &count,
                nullptr
            );

            std::vector<VkExtensionProperties> vk_extensions(count);
            VkResult vk_result = vkEnumerateDeviceExtensionProperties(
                handle(),
                layer_name_cstr,
                &count,
                vk_extensions.data()
            );
            if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
            {
                throw Error(vk_result);
            }

            std::vector<ExtensionProperties> extensions;
            extensions.reserve(vk_extensions.size());
            for (const auto& vk_ext : vk_extensions)
            {
                extensions.push_back(ExtensionProperties_from_vk(vk_ext));
            }
            return extensions;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to fetch available device extensions: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    FormatProperties PhysicalDevice::fetch_format_properties(
        VkFormat format
    ) const
    {
        VkFormatProperties vk_properties;
        vkGetPhysicalDeviceFormatProperties(
            handle(),
            format,
            &vk_properties
        );
        return FormatProperties_from_vk(vk_properties);
    }

    std::optional<VkFormat> PhysicalDevice::find_supported_image_format(
        const std::vector<VkFormat>& candidates,
        VkImageTiling tiling,
        VkFormatFeatureFlags features
    ) const
    {
        for (VkFormat format : candidates)
        {
            auto props = fetch_format_properties(format);
            if (tiling == VK_IMAGE_TILING_LINEAR
                && (props.linear_tiling_features & features) == features)
            {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL
                && (props.optimal_tiling_features & features) == features)
            {
                return format;
            }
        }
        return std::nullopt;
    }

    ImageFormatProperties PhysicalDevice::fetch_image_format_properties(
        VkFormat format,
        VkImageType type,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkImageCreateFlags flags
    ) const
    {
        try
        {
            VkImageFormatProperties vk_properties;
            VkResult vk_result = vkGetPhysicalDeviceImageFormatProperties(
                handle(),
                format,
                type,
                tiling,
                usage,
                flags,
                &vk_properties
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return ImageFormatProperties_from_vk(vk_properties);
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to fetch image format properties: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::optional<SwapchainSupport> PhysicalDevice::fetch_swapchain_support(
        const SurfacePtr& surface
    ) const
    {
        if (surface == nullptr)
        {
            return std::nullopt;
        }

        try
        {
            // check for extension
            auto available_extensions = fetch_available_extensions();
            bool has_extension = false;
            for (const auto& ext : available_extensions)
            {
                if (ext.name == VK_KHR_SWAPCHAIN_EXTENSION_NAME)
                {
                    has_extension = true;
                    break;
                }
            }
            if (!has_extension)
            {
                return std::nullopt;
            }

            // make sure the surface is supported by the physical device and at
            // least one of its queue families.
            bool device_supports_surface = false;
            for (size_t i = 0; i < queue_families().size(); i++)
            {
                VkBool32 vk_surface_support = VK_FALSE;
                VkResult vk_result = vkGetPhysicalDeviceSurfaceSupportKHR(
                    handle(),
                    i,
                    surface->handle(),
                    &vk_surface_support
                );
                if (vk_result != VK_SUCCESS)
                {
                    throw Error(
                        "failed to check physical device's surface support",
                        vk_result,
                        false
                    );
                }
                if (vk_surface_support)
                {
                    device_supports_surface = true;
                    break;
                }
            }
            if (!device_supports_surface)
            {
                return std::nullopt;
            }

            VkSurfaceCapabilitiesKHR vk_capabilities;
            VkResult vk_result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                handle(),
                surface->handle(),
                &vk_capabilities
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(
                    "failed to get physical device surface capabilities",
                    vk_result,
                    false
                );
            }

            std::vector<SurfaceFormat> surface_formats;
            {
                uint32_t surface_format_count;
                vkGetPhysicalDeviceSurfaceFormatsKHR(
                    handle(),
                    surface->handle(),
                    &surface_format_count,
                    nullptr
                );

                std::vector<VkSurfaceFormatKHR> vk_surface_formats(
                    surface_format_count
                );
                vk_result = vkGetPhysicalDeviceSurfaceFormatsKHR(
                    handle(),
                    surface->handle(),
                    &surface_format_count,
                    vk_surface_formats.data()
                );
                if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
                {
                    throw Error(
                        "failed to get physical device surface formats",
                        vk_result,
                        false
                    );
                }

                surface_formats.reserve(vk_surface_formats.size());
                for (const auto& vk_surface_format : vk_surface_formats)
                {
                    surface_formats.push_back(
                        SurfaceFormat_from_vk(vk_surface_format)
                    );
                }
            }

            std::vector<VkPresentModeKHR> present_modes;
            {
                uint32_t present_mode_count;
                vkGetPhysicalDeviceSurfacePresentModesKHR(
                    handle(),
                    surface->handle(),
                    &present_mode_count,
                    nullptr
                );

                present_modes.resize(present_mode_count);
                vk_result = vkGetPhysicalDeviceSurfacePresentModesKHR(
                    handle(),
                    surface->handle(),
                    &present_mode_count,
                    present_modes.data()
                );
                if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
                {
                    throw Error(
                        "failed to get physical device surface present modes",
                        vk_result,
                        false
                    );
                }
            }

            return SwapchainSupport{
                .capabilities = SurfaceCapabilities_from_vk(vk_capabilities),
                .surface_formats = surface_formats,
                .present_modes = present_modes
            };
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to fetch swapchain support details: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::vector<uint32_t> PhysicalDevice::find_queue_family_indices(
        VkQueueFlags must_support,
        VkQueueFlags must_not_support,
        const SurfacePtr& must_support_surface,
        uint32_t min_queue_count
    ) const
    {
        try
        {
            std::vector<uint32_t> indices;
            for (size_t i = 0; i < queue_families().size(); i++)
            {
                const auto& fam = queue_families()[i];
                if ((fam.queue_flags & must_support) != must_support)
                {
                    continue;
                }
                if ((fam.queue_flags & must_not_support) != 0)
                {
                    continue;
                }
                if (must_support_surface != nullptr)
                {
                    VkBool32 vk_surface_support = VK_FALSE;
                    VkResult vk_result = vkGetPhysicalDeviceSurfaceSupportKHR(
                        handle(),
                        (uint32_t)i,
                        must_support_surface->handle(),
                        &vk_surface_support
                    );
                    if (vk_result != VK_SUCCESS)
                    {
                        throw Error(
                            "failed to check physical device's surface support",
                            vk_result,
                            false
                        );
                    }

                    if (!vk_surface_support)
                    {
                        continue;
                    }
                }
                if (min_queue_count > 0 && fam.queue_count < min_queue_count)
                {
                    continue;
                }

                indices.push_back((uint32_t)i);
            }
            return indices;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to find queue family indices: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    uint32_t PhysicalDevice::find_first_queue_family_index(
        VkQueueFlags must_support,
        VkQueueFlags must_not_support,
        const SurfacePtr& must_support_surface,
        uint32_t min_queue_count
    ) const
    {
        auto indices = find_queue_family_indices(
            must_support,
            must_not_support,
            must_support_surface,
            min_queue_count
        );
        if (indices.empty())
        {
            throw Error("no queue family meets the required criteria");
        }
        return indices[0];
    }

    PhysicalDevice::PhysicalDevice(
        VkPhysicalDevice handle,
        const PhysicalDeviceProperties& properties,
        const PhysicalDeviceFeatures& features,
        const PhysicalDeviceMemoryProperties& memory_properties,
        const std::vector<QueueFamily>& queue_families
    )
        : _handle(handle),
        _properties(properties),
        _features(features),
        _memory_properties(memory_properties),
        _queue_families(queue_families)
    {}

    Context::Context(Context&& other) noexcept
        : _config(std::move(other._config))
    {
        _config = other._config;
        other._config = {};

        _allocator = other._allocator;
        other._allocator = nullptr;

        _vk_allocator = other._vk_allocator;
        _vk_allocator.pUserData = _allocator.get();
        other._vk_allocator = VkAllocationCallbacks{};

        _vk_instance = other._vk_instance;
        other._vk_instance = nullptr;
    }

    ContextPtr Context::create(
        const ContextConfig& config,
        const AllocatorPtr& allocator
    )
    {
        try
        {
            ContextPtr c = std::make_shared<Context_public_ctor>(
                config,
                allocator
            );

            // allocation callbacks
            {
                c->_vk_allocator.pUserData = c->allocator().get();
                c->_vk_allocator.pfnAllocation = vk_allocation_callback;
                c->_vk_allocator.pfnReallocation = vk_reallocation_callback;
                c->_vk_allocator.pfnFree = vk_free_callback;
                c->_vk_allocator.pfnInternalAllocation =
                    vk_internal_allocation_notification;
                c->_vk_allocator.pfnInternalFree =
                    vk_internal_free_notification;
            }

            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

            if (c->config().will_enumerate_portability)
                create_info.flags |=
                VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

            VkApplicationInfo app_info{};
            {
                app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;

                app_info.pApplicationName = c->config().app_name.c_str();
                app_info.applicationVersion = c->config().app_version.encode();

                app_info.pEngineName = c->config().engine_name.c_str();
                app_info.engineVersion = c->config().engine_version.encode();

                app_info.apiVersion = VulkanApiVersion_encode(
                    c->config().vulkan_api_version
                );
            }
            create_info.pApplicationInfo = &app_info;

            std::vector<const char*> layers_cstr;
            {
                layers_cstr.reserve(
                    c->config().layers.size()
                );
                for (const auto& layer : c->config().layers)
                {
                    layers_cstr.push_back(layer.c_str());
                }

                create_info.enabledLayerCount =
                    (uint32_t)layers_cstr.size();
                create_info.ppEnabledLayerNames =
                    layers_cstr.data();
            }

            std::vector<const char*> extensions_cstr;
            {
                extensions_cstr.reserve(
                    c->config().extensions.size()
                );
                for (const auto& ext : c->config().extensions)
                {
                    extensions_cstr.push_back(ext.c_str());
                }

                create_info.enabledExtensionCount =
                    (uint32_t)extensions_cstr.size();
                create_info.ppEnabledExtensionNames =
                    extensions_cstr.data();
            }

            VkResult vk_result = vkCreateInstance(
                &create_info,
                c->vk_allocator_ptr(),
                &c->_vk_instance
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return c;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create instance for context: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::vector<LayerProperties> Context::fetch_available_layers()
    {
        try
        {
            uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(
                &count,
                nullptr
            );

            std::vector<VkLayerProperties> vk_layers(count);
            VkResult vk_result = vkEnumerateInstanceLayerProperties(
                &count,
                vk_layers.data()
            );
            if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
            {
                throw Error(vk_result);
            }

            std::vector<LayerProperties> layers;
            layers.reserve(vk_layers.size());
            for (const auto& vk_layer : vk_layers)
            {
                layers.push_back(LayerProperties_from_vk(vk_layer));
            }
            return layers;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to fetch available instance layers: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::vector<ExtensionProperties> Context::fetch_available_extensions(
        const std::string& layer_name
    )
    {
        try
        {
            const char* layer_name_cstr = nullptr;
            if (!layer_name.empty())
            {
                layer_name_cstr = layer_name.c_str();
            }

            uint32_t count = 0;
            vkEnumerateInstanceExtensionProperties(
                layer_name_cstr,
                &count,
                nullptr
            );

            std::vector<VkExtensionProperties> vk_extensions(count);
            VkResult vk_result = vkEnumerateInstanceExtensionProperties(
                layer_name_cstr,
                &count,
                vk_extensions.data()
            );
            if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
            {
                throw Error(vk_result);
            }

            std::vector<ExtensionProperties> extensions;
            extensions.reserve(vk_extensions.size());
            for (const auto& vk_ext : vk_extensions)
            {
                extensions.push_back(ExtensionProperties_from_vk(vk_ext));
            }
            return extensions;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to fetch available instance extensions: "
                + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Context::set_allocator(const AllocatorPtr& allocator)
    {
        _allocator = allocator;
        _vk_allocator.pUserData = _allocator.get();
    }

    const VkAllocationCallbacks* Context::vk_allocator_ptr() const
    {
        if (_allocator == nullptr)
        {
            return nullptr;
        }
        return &_vk_allocator;
    }

    std::vector<PhysicalDevice> Context::fetch_physical_devices() const
    {
        try
        {
            uint32_t count = 0;
            vkEnumeratePhysicalDevices(_vk_instance, &count, nullptr);

            std::vector<VkPhysicalDevice> vk_physical_devices(count);
            VkResult vk_result = vkEnumeratePhysicalDevices(
                _vk_instance,
                &count,
                vk_physical_devices.data()
            );
            if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
            {
                throw Error(vk_result);
            }

            std::vector<PhysicalDevice> physical_devices;
            physical_devices.reserve(vk_physical_devices.size());
            for (const auto& vk_physical_device : vk_physical_devices)
            {
                VkPhysicalDeviceProperties vk_properties;
                vkGetPhysicalDeviceProperties(
                    vk_physical_device,
                    &vk_properties
                );
                auto properties = PhysicalDeviceProperties_from_vk(
                    vk_properties
                );

                VkPhysicalDeviceFeatures vk_features;
                vkGetPhysicalDeviceFeatures(vk_physical_device, &vk_features);
                auto features = PhysicalDeviceFeatures_from_vk(vk_features);

                VkPhysicalDeviceMemoryProperties vk_memory_properties;
                vkGetPhysicalDeviceMemoryProperties(
                    vk_physical_device,
                    &vk_memory_properties
                );
                auto memory_properties = PhysicalDeviceMemoryProperties_from_vk(
                    vk_memory_properties
                );

                uint32_t queue_family_count = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(
                    vk_physical_device,
                    &queue_family_count,
                    nullptr
                );

                std::vector<VkQueueFamilyProperties> vk_queue_families(
                    queue_family_count
                );
                vkGetPhysicalDeviceQueueFamilyProperties(
                    vk_physical_device,
                    &queue_family_count,
                    vk_queue_families.data()
                );

                std::vector<QueueFamily> queue_families;
                queue_families.reserve(vk_queue_families.size());
                for (const auto& vk_family : vk_queue_families)
                {
                    queue_families.push_back(QueueFamily_from_vk(vk_family));
                }

                physical_devices.push_back(PhysicalDevice(
                    vk_physical_device,
                    properties,
                    features,
                    memory_properties,
                    queue_families
                ));
            }

            return physical_devices;

        }
        catch (const Error& e)
        {
            throw Error(
                "failed to fetch physical devices: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Context::~Context()
    {
        vkDestroyInstance(_vk_instance, vk_allocator_ptr());
    }

    Context::Context(
        const ContextConfig& config,
        const AllocatorPtr& allocator
    )
        : _config(config), _allocator(allocator)
    {}

    DebugMessengerPtr DebugMessenger::create(
        const ContextPtr& context,
        VkDebugUtilsMessageSeverityFlagsEXT message_severity_filter,
        VkDebugUtilsMessageTypeFlagsEXT message_type_filter,
        const DebugCallback& callback
    )
    {
        try
        {
            DebugMessengerPtr messenger =
                std::make_shared<DebugMessenger_public_ctor>(
                    context,
                    message_severity_filter,
                    message_type_filter,
                    callback
                );

            VkDebugUtilsMessengerCreateInfoEXT create_info{
                .sType =
                VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,

                .pNext = nullptr,
                .flags = 0,
                .messageSeverity = messenger->message_severity_filter(),
                .messageType = messenger->message_type_filter(),
                .pfnUserCallback = vk_debug_callback,
                .pUserData = messenger.get()
            };

            VkResult vk_result = CreateDebugUtilsMessengerEXT(
                context->vk_instance(),
                &create_info,
                context->vk_allocator_ptr(),
                &messenger->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return messenger;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create debug messenger: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    DebugMessenger::~DebugMessenger()
    {
        _BV_LOCK_WPTR_OR_RETURN(context(), context_locked);
        DestroyDebugUtilsMessengerEXT(
            context_locked->vk_instance(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    DebugMessenger::DebugMessenger(
        const ContextPtr& context,
        VkDebugUtilsMessageSeverityFlagsEXT message_severity_filter,
        VkDebugUtilsMessageTypeFlagsEXT message_type_filter,
        const DebugCallback& callback
    )
        : _context(context),
        _message_severity_filter(message_severity_filter),
        _message_type_filter(message_type_filter),
        _callback(callback)
    {}

    SurfacePtr Surface::create(
        const ContextPtr& context,
        VkSurfaceKHR handle
    )
    {
        return std::make_shared<Surface_public_ctor>(context, handle);
    }

    Surface::~Surface()
    {
        _BV_LOCK_WPTR_OR_RETURN(context(), context_locked);
        vkDestroySurfaceKHR(
            context_locked->vk_instance(),
            _handle,
            context_locked->vk_allocator_ptr()
        );
    }

    Surface::Surface(
        const ContextPtr& context,
        VkSurfaceKHR handle
    )
        : _context(context), _handle(handle)
    {}

    void Queue::submit(
        const std::vector<VkPipelineStageFlags>& wait_stages,
        const std::vector<SemaphorePtr>& wait_semaphores,
        const std::vector<CommandBufferPtr>& command_buffers,
        const std::vector<SemaphorePtr>& signal_semaphores,
        const FencePtr& signal_fence
    )
    {
        try
        {
            std::vector<VkSemaphore> vk_semaphores(
                wait_semaphores.size() + signal_semaphores.size()
            );
            for (size_t i = 0; i < wait_semaphores.size(); i++)
            {
                vk_semaphores[i] = wait_semaphores[i]->handle();
            }
            for (size_t i = 0; i < signal_semaphores.size(); i++)
            {
                vk_semaphores[wait_semaphores.size() + i] =
                    signal_semaphores[i]->handle();
            }

            std::vector<VkCommandBuffer> vk_command_buffers(
                command_buffers.size()
            );
            for (size_t i = 0; i < command_buffers.size(); i++)
            {
                vk_command_buffers[i] = command_buffers[i]->handle();
            }

            VkSubmitInfo submit_info{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreCount = (uint32_t)wait_semaphores.size(),
                .pWaitSemaphores = vk_semaphores.data(),
                .pWaitDstStageMask = wait_stages.data(),
                .commandBufferCount = (uint32_t)vk_command_buffers.size(),
                .pCommandBuffers = vk_command_buffers.data(),
                .signalSemaphoreCount = (uint32_t)signal_semaphores.size(),

                .pSignalSemaphores =
                signal_semaphores.empty()
                ? nullptr : vk_semaphores.data() + wait_semaphores.size()
            };

            VkResult vk_result = vkQueueSubmit(
                handle(),
                1,
                &submit_info,
                signal_fence == nullptr ? nullptr : signal_fence->handle()
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to submit command buffer(s) to queue: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Queue::present(
        const std::vector<SemaphorePtr>& wait_semaphores,
        const SwapchainPtr& swapchain,
        uint32_t image_index,
        VkResult* out_vk_result
    )
    {
        try
        {
            std::vector<VkSemaphore> vk_semaphores(wait_semaphores.size());
            for (size_t i = 0; i < wait_semaphores.size(); i++)
            {
                vk_semaphores[i] = wait_semaphores[i]->handle();
            }

            VkSwapchainKHR vk_swapchain = swapchain->handle();

            VkPresentInfoKHR present_info{
                .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                .pNext = nullptr,
                .waitSemaphoreCount = (uint32_t)wait_semaphores.size(),
                .pWaitSemaphores = vk_semaphores.data(),
                .swapchainCount = 1,
                .pSwapchains = &vk_swapchain,
                .pImageIndices = &image_index,
                .pResults = nullptr
            };

            VkResult vk_result = vkQueuePresentKHR(
                handle(),
                &present_info
            );
            if (out_vk_result != nullptr)
            {
                *out_vk_result = vk_result;
            }
            if (vk_result != VK_SUCCESS && vk_result != VK_SUBOPTIMAL_KHR)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to queue image for presentation: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Queue::wait_idle()
    {
        VkResult vk_result = vkQueueWaitIdle(handle());
        if (vk_result != VK_SUCCESS)
        {
            throw Error(
                "failed to wait for queue to become idle",
                vk_result,
                false
            );
        }
    }

    Queue::Queue(
        const DeviceWPtr& device,
        uint32_t queue_family_index,
        uint32_t queue_index,
        VkQueue handle
    )
        : _device(device),
        _queue_family_index(queue_family_index),
        _queue_index(queue_index),
        _handle(handle)
    {}

    DevicePtr Device::create(
        const ContextPtr& context,
        const PhysicalDevice& physical_device,
        const DeviceConfig& config
    )
    {
        try
        {
            DevicePtr device = std::make_shared<Device_public_ctor>(
                context,
                physical_device,
                config
            );

            std::vector<VkDeviceQueueCreateInfo> vk_queue_requests;
            std::vector<std::vector<float>> wastes_priorities(
                device->config().queue_requests.size()
            );
            for (size_t i = 0; i < device->config().queue_requests.size(); i++)
            {
                const auto& queue_request = device->config().queue_requests[i];

                if (queue_request.priorities.size()
                    != queue_request.num_queues_to_create)
                {
                    throw Error(
                        "there should be the same number of queue priorities "
                        "as the number of queues to create"
                    );
                }

                vk_queue_requests.push_back(
                    QueueRequest_to_vk(
                        queue_request,
                        wastes_priorities[i]
                    )
                );
            }

            std::vector<const char*> layers_cstr;
            {
                layers_cstr.reserve(
                    context->config().layers.size()
                );
                for (const auto& layer
                    : context->config().layers)
                {
                    layers_cstr.push_back(layer.c_str());
                }
            }

            std::vector<const char*> extensions_cstr;
            {
                extensions_cstr.reserve(
                    device->config().extensions.size()
                );
                for (const auto& ext : device->config().extensions)
                {
                    extensions_cstr.push_back(ext.c_str());
                }
            }

            VkPhysicalDeviceFeatures vk_enabled_features =
                PhysicalDeviceFeatures_to_vk(
                    device->config().enabled_features
                );

            VkDeviceCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .queueCreateInfoCount = (uint32_t)vk_queue_requests.size(),
                .pQueueCreateInfos = vk_queue_requests.data(),
                .enabledLayerCount = (uint32_t)layers_cstr.size(),
                .ppEnabledLayerNames = layers_cstr.data(),
                .enabledExtensionCount = (uint32_t)extensions_cstr.size(),
                .ppEnabledExtensionNames = extensions_cstr.data(),
                .pEnabledFeatures = &vk_enabled_features
            };

            VkResult vk_result = vkCreateDevice(
                device->physical_device().handle(),
                &create_info,
                context->vk_allocator_ptr(),
                &device->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return device;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create device: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    QueuePtr Device::retrieve_queue(
        const DevicePtr& device,
        uint32_t queue_family_index,
        uint32_t queue_index
    )
    {
        VkQueue vk_queue;
        vkGetDeviceQueue(
            device->handle(),
            queue_family_index,
            queue_index,
            &vk_queue
        );
        return std::make_shared<Queue_public_ctor>(
            device,
            queue_family_index,
            queue_index,
            vk_queue
        );
    }

    void Device::wait_idle()
    {
        VkResult vk_result = vkDeviceWaitIdle(_handle);
        if (vk_result != VK_SUCCESS)
        {
            throw Error(
                "failed to wait for device to become idle",
                vk_result,
                false
            );
        }
    }

    Device::~Device()
    {
        _BV_LOCK_WPTR_OR_RETURN(context(), context_locked);
        vkDestroyDevice(handle(), context_locked->vk_allocator_ptr());
    }

    Device::Device(
        const ContextPtr& context,
        const PhysicalDevice& physical_device,
        const DeviceConfig& config
    )
        : _context(context),
        _physical_device(physical_device),
        _config(config)
    {}

    ImagePtr Image::create(
        const DevicePtr& device,
        const ImageConfig& config
    )
    {
        try
        {
            ImageFormatProperties img_format_props{};
            try
            {
                img_format_props =
                    device->physical_device().fetch_image_format_properties(
                        config.format,
                        config.image_type,
                        config.tiling,
                        config.usage,
                        config.flags
                    );
            }
            catch (const Error& e)
            {
                if (e.vk_result().has_value()
                    && e.vk_result().value() == VK_ERROR_FORMAT_NOT_SUPPORTED)
                {
                    throw Error(
                        "image format not supported with the provided "
                        "parameters",
                        e.vk_result(),
                        true
                    );
                }
                else
                {
                    throw e;
                }
            }

            ImagePtr img = std::make_shared<Image_public_ctor>(
                device,
                config
            );

            VkImageCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .pNext = nullptr,
                .flags = img->config().flags,
                .imageType = img->config().image_type,
                .format = img->config().format,
                .extent = Extent3d_to_vk(img->config().extent),
                .mipLevels = img->config().mip_levels,
                .arrayLayers = img->config().array_layers,
                .samples = img->config().samples,
                .tiling = img->config().tiling,
                .usage = img->config().usage,
                .sharingMode = img->config().sharing_mode,

                .queueFamilyIndexCount =
                (uint32_t)img->config().queue_family_indices.size(),

                .pQueueFamilyIndices =
                img->config().queue_family_indices.data(),

                .initialLayout = img->config().initial_layout
            };

            VkResult vk_result = vkCreateImage(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &img->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }

            VkMemoryRequirements vk_mem_requirements;
            vkGetImageMemoryRequirements(
                device->handle(),
                img->handle(),
                &vk_mem_requirements
            );
            img->_memory_requirements = MemoryRequirements_from_vk(
                vk_mem_requirements
            );

            return img;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create image: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Image::bind_memory(
        const DeviceMemoryPtr& memory,
        VkDeviceSize memory_offset
    )
    {
        try
        {
            VkResult vk_result = vkBindImageMemory(
                lock_wptr(device())->handle(),
                handle(),
                memory->handle(),
                memory_offset
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to bind image memory: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Image::~Image()
    {
        if (created_externally())
        {
            return;
        }

        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyImage(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Image::Image(const DevicePtr& device, const ImageConfig& config)
        : _created_externally(false),
        _device(device),
        _config(config),
        _memory_requirements({}),
        _handle(nullptr)
    {}

    Image::Image(VkImage handle_created_externally)
        : _created_externally(true),
        _device({}),
        _config({}),
        _memory_requirements({}),
        _handle(handle_created_externally)
    {}

    SwapchainPtr Swapchain::create(
        const DevicePtr& device,
        const SurfacePtr& surface,
        const SwapchainConfig& config,
        const SwapchainPtr& old_swapchain
    )
    {
        try
        {
            SwapchainPtr sc = std::make_shared<Swapchain_public_ctor>(
                device,
                surface,
                config,
                old_swapchain
            );

            VkSwapchainKHR vk_old_swapchain = nullptr;
            if (old_swapchain != nullptr)
            {
                vk_old_swapchain = old_swapchain->handle();
            }

            VkSwapchainCreateInfoKHR create_info{
                .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                .pNext = nullptr,
                .flags = sc->config().flags,
                .surface = surface->handle(),
                .minImageCount = sc->config().min_image_count,
                .imageFormat = sc->config().image_format,
                .imageColorSpace = sc->config().image_color_space,
                .imageExtent = Extent2d_to_vk(sc->config().image_extent),
                .imageArrayLayers = sc->config().image_array_layers,
                .imageUsage = sc->config().image_usage,
                .imageSharingMode = sc->config().image_sharing_mode,

                .queueFamilyIndexCount =
                (uint32_t)sc->config().queue_family_indices.size(),

                .pQueueFamilyIndices = sc->config().queue_family_indices.data(),
                .preTransform = sc->config().pre_transform,
                .compositeAlpha = sc->config().composite_alpha,
                .presentMode = sc->config().present_mode,
                .clipped = sc->config().clipped,
                .oldSwapchain = vk_old_swapchain
            };

            VkResult vk_result = vkCreateSwapchainKHR(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &sc->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }

            uint32_t actual_image_count;
            vkGetSwapchainImagesKHR(
                device->handle(),
                sc->handle(),
                &actual_image_count,
                nullptr
            );

            std::vector<VkImage> vk_images(actual_image_count);
            vk_result = vkGetSwapchainImagesKHR(
                device->handle(),
                sc->handle(),
                &actual_image_count,
                vk_images.data()
            );
            if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
            {
                throw Error(
                    "failed to retrieve swapchain images after creating it",
                    vk_result,
                    false
                );
            }

            sc->_images.reserve(actual_image_count);
            for (auto vk_image : vk_images)
            {
                sc->_images.push_back(
                    std::make_shared<Image_public_ctor>(vk_image)
                );
            }

            return sc;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create swapchain: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    uint32_t Swapchain::acquire_next_image(
        const SemaphorePtr& semaphore,
        const FencePtr& fence,
        uint64_t timeout,
        VkResult* out_vk_result
    )
    {
        try
        {
            uint32_t image_index;
            VkResult vk_result = vkAcquireNextImageKHR(
                lock_wptr(device())->handle(),
                handle(),
                timeout,
                semaphore == nullptr ? nullptr : semaphore->handle(),
                fence == nullptr ? nullptr : fence->handle(),
                &image_index
            );
            if (out_vk_result != nullptr)
            {
                *out_vk_result = vk_result;
            }
            if (vk_result == VK_SUCCESS
                || vk_result == VK_SUBOPTIMAL_KHR)
            {
                return image_index;
            }
            throw Error(vk_result);
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to acquire next swapchain image: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Swapchain::~Swapchain()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroySwapchainKHR(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Swapchain::Swapchain(
        const DevicePtr& device,
        const SurfacePtr& surface,
        const SwapchainConfig& config,
        const SwapchainPtr& old_swapchain
    )
        : _device(device),
        _surface(surface),
        _config(config),
        _old_swapchain(std::nullopt)
    {
        if (old_swapchain != nullptr)
        {
            _old_swapchain = old_swapchain;
        }
    }

    ImageViewPtr ImageView::create(
        const DevicePtr& device,
        const ImagePtr& image,
        const ImageViewConfig& config
    )
    {
        try
        {
            ImageViewPtr view = std::make_shared<ImageView_public_ctor>(
                device,
                image,
                config
            );

            VkImageViewCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = view->config().flags,
                .image = lock_wptr(view->image())->handle(),
                .viewType = view->config().view_type,
                .format = view->config().format,
                .components = ComponentMapping_to_vk(view->config().components),
                .subresourceRange = ImageSubresourceRange_to_vk(
                    view->config().subresource_range
                )
            };

            VkResult vk_result = vkCreateImageView(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &view->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return view;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create image view: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    ImageView::~ImageView()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyImageView(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    ImageView::ImageView(
        const DevicePtr& device,
        const ImagePtr& image,
        const ImageViewConfig& config
    )
        : _device(device),
        _image(image),
        _config(config)
    {}

    ShaderModulePtr ShaderModule::create(
        const DevicePtr& device,
        const std::vector<uint8_t>& code
    )
    {
        try
        {
            ShaderModulePtr module = std::make_shared<ShaderModule_public_ctor>(
                device
            );

            std::vector<uint8_t> code_aligned;
            bool needs_alignment = (code.size() % 8 != 0);
            if (needs_alignment)
            {
                code_aligned = code;
                for (size_t i = 0; i < 8 - (code.size() % 8); i++)
                {
                    code_aligned.push_back(0);
                }
            }

            VkShaderModuleCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .codeSize = code.size(),
                .pCode = reinterpret_cast<const uint32_t*>(
                    needs_alignment ? code_aligned.data() : code.data()
                    )
            };

            VkResult vk_result = vkCreateShaderModule(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &module->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return module;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create shader module: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    ShaderModule::~ShaderModule()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyShaderModule(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    ShaderModule::ShaderModule(const DevicePtr& device)
        : _device(device)
    {}

    SamplerPtr Sampler::create(
        const DevicePtr& device,
        const SamplerConfig& config
    )
    {
        try
        {
            SamplerPtr sampler = std::make_shared<Sampler_public_ctor>(
                device,
                config
            );

            VkSamplerCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .pNext = nullptr,
                .flags = sampler->config().flags,
                .magFilter = sampler->config().mag_filter,
                .minFilter = sampler->config().min_filter,
                .mipmapMode = sampler->config().mipmap_mode,
                .addressModeU = sampler->config().address_mode_u,
                .addressModeV = sampler->config().address_mode_v,
                .addressModeW = sampler->config().address_mode_w,
                .mipLodBias = sampler->config().mip_lod_bias,
                .anisotropyEnable = sampler->config().anisotropy_enable,
                .maxAnisotropy = sampler->config().max_anisotropy,
                .compareEnable = sampler->config().compare_enable,
                .compareOp = sampler->config().compare_op,
                .minLod = sampler->config().min_lod,
                .maxLod = sampler->config().max_lod,
                .borderColor = sampler->config().border_color,

                .unnormalizedCoordinates =
                sampler->config().unnormalized_coordinates
            };

            VkResult vk_result = vkCreateSampler(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &sampler->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return sampler;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create sampler: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Sampler::~Sampler()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroySampler(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Sampler::Sampler(
        const DevicePtr& device,
        const SamplerConfig& config
    )
        : _device(device),
        _config(config)
    {}

    DescriptorSetLayoutPtr DescriptorSetLayout::create(
        const DevicePtr& device,
        const DescriptorSetLayoutConfig& config
    )
    {
        try
        {
            DescriptorSetLayoutPtr layout =
                std::make_shared<DescriptorSetLayout_public_ctor>(
                    device,
                    config
                );

            std::vector<VkDescriptorSetLayoutBinding> vk_bindings(
                layout->config().bindings.size()
            );
            std::vector<std::vector<VkSampler>> wastes_vk_immutable_samplers(
                layout->config().bindings.size()
            );
            for (size_t i = 0; i < layout->config().bindings.size(); i++)
            {
                vk_bindings[i] = DescriptorSetLayoutBinding_to_vk(
                    layout->config().bindings[i],
                    wastes_vk_immutable_samplers[i]
                );
            }

            VkDescriptorSetLayoutCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = layout->config().flags,
                .bindingCount = (uint32_t)vk_bindings.size(),
                .pBindings = vk_bindings.data()
            };

            VkResult vk_result = vkCreateDescriptorSetLayout(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &layout->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return layout;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create descriptor set layout: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    DescriptorSetLayout::~DescriptorSetLayout()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyDescriptorSetLayout(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    DescriptorSetLayout::DescriptorSetLayout(
        const DevicePtr& device,
        const DescriptorSetLayoutConfig& config
    )
        : _device(device),
        _config(config)
    {}

    PipelineLayoutPtr PipelineLayout::create(
        const DevicePtr& device,
        const PipelineLayoutConfig& config
    )
    {
        try
        {
            PipelineLayoutPtr layout =
                std::make_shared<PipelineLayout_public_ctor>(
                    device,
                    config
                );

            std::vector<VkDescriptorSetLayout> vk_set_layouts(
                layout->config().set_layouts.size()
            );
            for (size_t i = 0; i < layout->config().set_layouts.size(); i++)
            {
                vk_set_layouts[i] =
                    lock_wptr(layout->config().set_layouts[i])->handle();
            }

            std::vector<VkPushConstantRange> vk_push_constant_ranges(
                layout->config().push_constant_ranges.size()
            );
            for (size_t i = 0;
                i < layout->config().push_constant_ranges.size();
                i++)
            {
                vk_push_constant_ranges[i] = PushConstantRange_to_vk(
                    layout->config().push_constant_ranges[i]
                );
            }

            VkPipelineLayoutCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = layout->config().flags,
                .setLayoutCount = (uint32_t)vk_set_layouts.size(),
                .pSetLayouts = vk_set_layouts.data(),

                .pushConstantRangeCount =
                (uint32_t)vk_push_constant_ranges.size(),

                .pPushConstantRanges = vk_push_constant_ranges.data()
            };

            VkResult vk_result = vkCreatePipelineLayout(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &layout->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return layout;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create pipeline layout: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    PipelineLayout::~PipelineLayout()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyPipelineLayout(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    PipelineLayout::PipelineLayout(
        const DevicePtr& device,
        const PipelineLayoutConfig& config
    )
        : _device(device), _config(config)
    {}

    RenderPassPtr RenderPass::create(
        const DevicePtr& device,
        const RenderPassConfig& config
    )
    {
        try
        {
            RenderPassPtr pass = std::make_shared<RenderPass_public_ctor>(
                device,
                config
            );

            std::vector<VkAttachmentDescription> vk_attachments(
                pass->config().attachments.size()
            );
            for (size_t i = 0; i < pass->config().attachments.size(); i++)
            {
                vk_attachments[i] = Attachment_to_vk(
                    pass->config().attachments[i]
                );
            }

            std::vector<VkSubpassDescription> vk_subpasses(
                pass->config().subpasses.size()
            );
            std::vector<std::vector<VkAttachmentReference>>
                wastes_vk_input_attachments(pass->config().subpasses.size());
            std::vector<std::vector<VkAttachmentReference>>
                wastes_vk_color_attachments(pass->config().subpasses.size());
            std::vector<std::vector<VkAttachmentReference>>
                wastes_vk_resolve_attachments(pass->config().subpasses.size());
            std::vector<VkAttachmentReference>
                wastes_vk_depth_stencil_attachment(
                    pass->config().subpasses.size()
                );
            std::vector<std::vector<uint32_t>>
                wastes_preserve_attachment_indices(
                    pass->config().subpasses.size()
                );
            for (size_t i = 0; i < pass->config().subpasses.size(); i++)
            {
                vk_subpasses[i] = Subpass_to_vk(
                    pass->config().subpasses[i],
                    wastes_vk_input_attachments[i],
                    wastes_vk_color_attachments[i],
                    wastes_vk_resolve_attachments[i],
                    wastes_vk_depth_stencil_attachment[i],
                    wastes_preserve_attachment_indices[i]
                );
            }

            std::vector<VkSubpassDependency> vk_dependencies(
                pass->config().dependencies.size()
            );
            for (size_t i = 0; i < pass->config().dependencies.size(); i++)
            {
                vk_dependencies[i] = SubpassDependency_to_vk(
                    pass->config().dependencies[i]
                );
            }

            VkRenderPassCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .pNext = nullptr,
                .flags = pass->config().flags,
                .attachmentCount = (uint32_t)vk_attachments.size(),
                .pAttachments = vk_attachments.data(),
                .subpassCount = (uint32_t)vk_subpasses.size(),
                .pSubpasses = vk_subpasses.data(),
                .dependencyCount = (uint32_t)vk_dependencies.size(),
                .pDependencies = vk_dependencies.data()
            };

            VkResult vk_result = vkCreateRenderPass(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &pass->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return pass;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create render pass: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    RenderPass::~RenderPass()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyRenderPass(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    RenderPass::RenderPass(
        const DevicePtr& device,
        const RenderPassConfig& config
    )
        : _device(device), _config(config)
    {}

    GraphicsPipelinePtr GraphicsPipeline::create(
        const DevicePtr& device,
        const GraphicsPipelineConfig& config,
        const PipelineCachePtr& cache
    )
    {
        try
        {
            GraphicsPipelinePtr pipe =
                std::make_shared<GraphicsPipeline_public_ctor>(
                    device,
                    config,
                    cache
                );

            std::vector<VkPipelineShaderStageCreateInfo> vk_stages(
                pipe->config().stages.size()
            );
            std::vector<VkSpecializationInfo> wastes_vk_specialization_info(
                pipe->config().stages.size()
            );
            std::vector<std::vector<VkSpecializationMapEntry>>
                wastes_vk_map_entries(pipe->config().stages.size());
            std::vector<std::vector<uint8_t>> wastes_data(
                pipe->config().stages.size()
            );
            for (size_t i = 0; i < pipe->config().stages.size(); i++)
            {
                vk_stages[i] = ShaderStage_to_vk(
                    pipe->config().stages[i],
                    wastes_vk_specialization_info[i],
                    wastes_vk_map_entries[i],
                    wastes_data[i]
                );
            }

            VkPipelineVertexInputStateCreateInfo vk_vertex_input_state{};
            std::vector<VkVertexInputBindingDescription>
                waste_vk_binding_descriptions;
            std::vector<VkVertexInputAttributeDescription>
                waste_vk_attribute_descriptions;
            if (pipe->config().vertex_input_state.has_value())
            {
                vk_vertex_input_state = VertexInputState_to_vk(
                    pipe->config().vertex_input_state.value(),
                    waste_vk_binding_descriptions,
                    waste_vk_attribute_descriptions
                );
            }

            VkPipelineInputAssemblyStateCreateInfo vk_input_assembly_state{};
            if (pipe->config().input_assembly_state.has_value())
            {
                vk_input_assembly_state = InputAssemblyState_to_vk(
                    pipe->config().input_assembly_state.value()
                );
            }

            VkPipelineTessellationStateCreateInfo vk_tessellation_state{};
            if (pipe->config().tessellation_state.has_value())
            {
                vk_tessellation_state = TessellationState_to_vk(
                    pipe->config().tessellation_state.value()
                );
            }

            VkPipelineViewportStateCreateInfo vk_viewport_state{};
            std::vector<VkViewport> waste_vk_viewports;
            std::vector<VkRect2D> waste_vk_scissors;
            if (pipe->config().viewport_state.has_value())
            {
                vk_viewport_state = ViewportState_to_vk(
                    pipe->config().viewport_state.value(),
                    waste_vk_viewports,
                    waste_vk_scissors
                );
            }

            VkPipelineRasterizationStateCreateInfo vk_rasterization_state{};
            if (pipe->config().rasterization_state.has_value())
            {
                vk_rasterization_state = RasterizationState_to_vk(
                    pipe->config().rasterization_state.value()
                );
            }

            VkPipelineMultisampleStateCreateInfo vk_multisample_state{};
            std::vector<VkSampleMask> waste_sample_mask;
            if (pipe->config().multisample_state.has_value())
            {
                vk_multisample_state = MultisampleState_to_vk(
                    pipe->config().multisample_state.value(),
                    waste_sample_mask
                );
            }

            VkPipelineDepthStencilStateCreateInfo vk_depth_stencil_state{};
            if (pipe->config().depth_stencil_state.has_value())
            {
                vk_depth_stencil_state = DepthStencilState_to_vk(
                    pipe->config().depth_stencil_state.value()
                );
            }

            VkPipelineColorBlendStateCreateInfo vk_color_blend_state{};
            std::vector<VkPipelineColorBlendAttachmentState>
                waste_vk_color_blend_attachments;
            if (pipe->config().color_blend_state.has_value())
            {
                vk_color_blend_state = ColorBlendState_to_vk(
                    pipe->config().color_blend_state.value(),
                    waste_vk_color_blend_attachments
                );
            }

            std::vector<VkDynamicState> waste_dynamic_states;
            VkPipelineDynamicStateCreateInfo vk_dynamic_states =
                DynamicStates_to_vk(
                    pipe->config().dynamic_states,
                    waste_dynamic_states
                );

            VkGraphicsPipelineCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = pipe->config().flags,
                .stageCount = (uint32_t)vk_stages.size(),
                .pStages = vk_stages.data(),

                .pVertexInputState =
                pipe->config().vertex_input_state.has_value()
                ? &vk_vertex_input_state : nullptr,

                .pInputAssemblyState =
                pipe->config().input_assembly_state.has_value()
                ? &vk_input_assembly_state : nullptr,

                .pTessellationState =
                pipe->config().tessellation_state.has_value()
                ? &vk_tessellation_state : nullptr,

                .pViewportState =
                pipe->config().viewport_state.has_value()
                ? &vk_viewport_state : nullptr,

                .pRasterizationState =
                pipe->config().rasterization_state.has_value()
                ? &vk_rasterization_state : nullptr,

                .pMultisampleState =
                pipe->config().multisample_state.has_value()
                ? &vk_multisample_state : nullptr,

                .pDepthStencilState =
                pipe->config().depth_stencil_state.has_value()
                ? &vk_depth_stencil_state : nullptr,

                .pColorBlendState =
                pipe->config().color_blend_state.has_value()
                ? &vk_color_blend_state : nullptr,

                .pDynamicState =
                pipe->config().dynamic_states.empty()
                ? nullptr : &vk_dynamic_states,

                .layout = lock_wptr(pipe->config().layout)->handle(),
                .renderPass = lock_wptr(pipe->config().render_pass)->handle(),
                .subpass = pipe->config().subpass_index,

                .basePipelineHandle =
                pipe->config().base_pipeline.has_value()
                ? lock_wptr(pipe->config().base_pipeline.value())->handle()
                : nullptr,

                .basePipelineIndex = -1
            };

            VkResult vk_result = vkCreateGraphicsPipelines(
                device->handle(),
                cache == nullptr ? nullptr : cache->handle(),
                1,
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &pipe->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return pipe;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create graphics pipeline: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    GraphicsPipeline::~GraphicsPipeline()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyPipeline(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    GraphicsPipeline::GraphicsPipeline(
        const DevicePtr& device,
        const GraphicsPipelineConfig& config,
        const PipelineCachePtr& cache
    )
        : _device(device), _config(config), _cache(cache)
    {}

    ComputePipelinePtr ComputePipeline::create(
        const DevicePtr& device,
        const ComputePipelineConfig& config,
        const PipelineCachePtr& cache
    )
    {
        try
        {
            ComputePipelinePtr pipe =
                std::make_shared<ComputePipeline_public_ctor>(
                    device,
                    config,
                    cache
                );

            VkSpecializationInfo waste_vk_specialization_info;
            std::vector<VkSpecializationMapEntry> waste_vk_map_entries;
            std::vector<uint8_t> waste_data;
            VkPipelineShaderStageCreateInfo vk_stage = ShaderStage_to_vk(
                pipe->config().stage,
                waste_vk_specialization_info,
                waste_vk_map_entries,
                waste_data
            );

            VkComputePipelineCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = pipe->config().flags,
                .stage = vk_stage,
                .layout = lock_wptr(pipe->config().layout)->handle(),

                .basePipelineHandle =
                pipe->config().base_pipeline.has_value()
                ? lock_wptr(pipe->config().base_pipeline.value())->handle()
                : nullptr,

                .basePipelineIndex = -1
            };

            VkResult vk_result = vkCreateComputePipelines(
                device->handle(),
                cache == nullptr ? nullptr : cache->handle(),
                1,
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &pipe->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return pipe;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create compute pipeline: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    ComputePipeline::~ComputePipeline()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyPipeline(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    ComputePipeline::ComputePipeline(
        const DevicePtr& device,
        const ComputePipelineConfig& config,
        const PipelineCachePtr& cache
    )
        : _device(device), _config(config), _cache(cache)
    {}

    FramebufferPtr Framebuffer::create(
        const DevicePtr& device,
        const FramebufferConfig& config
    )
    {
        try
        {
            FramebufferPtr buf = std::make_shared<Framebuffer_public_ctor>(
                device,
                config
            );

            std::vector<VkImageView> vk_attachments(
                buf->config().attachments.size()
            );
            for (size_t i = 0; i < buf->config().attachments.size(); i++)
            {
                vk_attachments[i] =
                    lock_wptr(buf->config().attachments[i])->handle();
            }

            VkFramebufferCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = buf->config().flags,
                .renderPass = lock_wptr(buf->config().render_pass)->handle(),
                .attachmentCount = (uint32_t)vk_attachments.size(),
                .pAttachments = vk_attachments.data(),
                .width = buf->config().width,
                .height = buf->config().height,
                .layers = buf->config().layers
            };

            VkResult vk_result = vkCreateFramebuffer(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &buf->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return buf;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create framebuffer: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Framebuffer::~Framebuffer()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyFramebuffer(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Framebuffer::Framebuffer(
        const DevicePtr& device,
        const FramebufferConfig& config
    )
        : _device(device), _config(config)
    {}

    void CommandBuffer::reset(VkCommandBufferResetFlags flags)
    {
        VkResult vk_result = vkResetCommandBuffer(_handle, flags);
        if (vk_result != VK_SUCCESS)
        {
            throw Error(
                "failed to reset command buffer",
                vk_result,
                false
            );
        }
    }

    void CommandBuffer::begin(
        VkCommandBufferUsageFlags flags,
        std::optional<CommandBufferInheritance> inheritance
    )
    {
        try
        {
            VkCommandBufferInheritanceInfo vk_inheritance;
            if (inheritance.has_value())
            {
                vk_inheritance = VkCommandBufferInheritanceInfo{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
                    .pNext = nullptr,
                    .renderPass = lock_wptr(inheritance->render_pass)->handle(),
                    .subpass = inheritance->subpass_index,

                    .framebuffer =
                    inheritance->framebuffer.has_value()
                    ? lock_wptr(
                        inheritance->framebuffer.value()
                    )->handle()
                    : nullptr,

                    .occlusionQueryEnable = inheritance->occlusion_query_enable,
                    .queryFlags = inheritance->query_flags,
                    .pipelineStatistics = inheritance->pipeline_statistics
                };
            }

            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.pNext = nullptr;
            begin_info.flags = flags;
            begin_info.pInheritanceInfo =
                inheritance.has_value() ? &vk_inheritance : nullptr;

            VkResult vk_result = vkBeginCommandBuffer(handle(), &begin_info);
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to begin recording a command buffer: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void CommandBuffer::end()
    {
        VkResult vk_result = vkEndCommandBuffer(handle());
        if (vk_result != VK_SUCCESS)
        {
            throw Error(
                "failed to end recording a command buffer",
                vk_result,
                false
            );
        }
    }

    CommandBuffer::~CommandBuffer()
    {
        _BV_LOCK_WPTR_OR_RETURN(pool(), pool_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            pool_locked->device(),
            device_locked
        );

        vkFreeCommandBuffers(
            device_locked->handle(),
            pool_locked->handle(),
            1,
            &_handle
        );
    }

    CommandBuffer::CommandBuffer(
        const CommandPoolWPtr& pool,
        VkCommandBuffer handle
    )
        : _pool(pool),
        _handle(handle)
    {}

    CommandPoolPtr CommandPool::create(
        const DevicePtr& device,
        const CommandPoolConfig& config
    )
    {
        try
        {
            CommandPoolPtr pool = std::make_shared<CommandPool_public_ctor>(
                device,
                config
            );

            VkCommandPoolCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = pool->config().flags,
                .queueFamilyIndex = pool->config().queue_family_index
            };

            VkResult vk_result = vkCreateCommandPool(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &pool->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return pool;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create command pool: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    CommandBufferPtr CommandPool::allocate_buffer(
        const CommandPoolPtr& pool,
        VkCommandBufferLevel level
    )
    {
        try
        {
            VkCommandBufferAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = pool->handle(),
                .level = level,
                .commandBufferCount = 1
            };

            VkCommandBuffer vk_command_buffer;
            VkResult vk_result = vkAllocateCommandBuffers(
                lock_wptr(pool->device())->handle(),
                &alloc_info,
                &vk_command_buffer
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return std::make_shared<CommandBuffer_public_ctor>(
                pool,
                vk_command_buffer
            );
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to allocate command buffer: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::vector<CommandBufferPtr> CommandPool::allocate_buffers(
        const CommandPoolPtr& pool,
        VkCommandBufferLevel level,
        uint32_t count
    )
    {
        if (count < 1)
        {
            return {};
        }

        try
        {
            VkCommandBufferAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .pNext = nullptr,
                .commandPool = pool->handle(),
                .level = level,
                .commandBufferCount = count
            };

            std::vector<VkCommandBuffer> vk_command_buffers(count);
            VkResult vk_result = vkAllocateCommandBuffers(
                lock_wptr(pool->device())->handle(),
                &alloc_info,
                vk_command_buffers.data()
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }

            std::vector<CommandBufferPtr> command_buffers(count);
            for (size_t i = 0; i < count; i++)
            {
                command_buffers[i] =
                    std::make_shared<CommandBuffer_public_ctor>(
                        pool,
                        vk_command_buffers[i]
                    );
            }
            return command_buffers;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to allocate command buffer(s): " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    CommandPool::~CommandPool()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyCommandPool(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    CommandPool::CommandPool(
        const DevicePtr& device,
        const CommandPoolConfig& config
    )
        : _device(device), _config(config)
    {}

    SemaphorePtr Semaphore::create(const DevicePtr& device)
    {
        try
        {
            SemaphorePtr sema = std::make_shared<Semaphore_public_ctor>(device);

            VkSemaphoreCreateInfo  create_info{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0
            };

            VkResult vk_result = vkCreateSemaphore(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &sema->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return sema;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create semaphore: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Semaphore::~Semaphore()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroySemaphore(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Semaphore::Semaphore(const DevicePtr& device)
        : _device(device)
    {}

    FencePtr Fence::create(
        const DevicePtr& device,
        VkFenceCreateFlags flags
    )
    {
        try
        {
            FencePtr fence = std::make_shared<Fence_public_ctor>(device);

            VkFenceCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                .pNext = nullptr,
                .flags = flags
            };

            VkResult vk_result = vkCreateFence(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &fence->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return fence;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create fence: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Fence::wait(uint64_t timeout)
    {
        try
        {
            VkResult vk_result = vkWaitForFences(
                lock_wptr(device())->handle(),
                1,
                &_handle,
                VK_TRUE,
                timeout
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to wait for fence to become signaled: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Fence::wait_multiple(
        const std::vector<FencePtr>& fences,
        bool wait_all,
        uint64_t timeout
    )
    {
        if (fences.empty())
        {
            return;
        }

        try
        {
            std::vector<VkFence> vk_fences(fences.size());
            for (size_t i = 0; i < fences.size(); i++)
            {
                vk_fences[i] = fences[i]->handle();
            }

            VkResult vk_result = vkWaitForFences(
                lock_wptr(fences[0]->device())->handle(),
                (uint32_t)vk_fences.size(),
                vk_fences.data(),
                wait_all,
                timeout
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to wait for fence(s) to become signaled: "
                + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Fence::reset()
    {
        try
        {
            VkResult vk_result = vkResetFences(
                lock_wptr(device())->handle(),
                1,
                &_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to reset fence: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    bool Fence::is_signaled() const
    {
        try
        {
            VkResult vk_result = vkGetFenceStatus(
                lock_wptr(device())->handle(),
                handle()
            );
            if (vk_result == VK_SUCCESS)
            {
                return true;
            }
            else if (vk_result == VK_NOT_READY)
            {
                return false;
            }
            throw Error(vk_result);
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to get fence status: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Fence::~Fence()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyFence(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Fence::Fence(const DevicePtr& device)
        : _device(device)
    {}

    BufferPtr Buffer::create(
        const DevicePtr& device,
        const BufferConfig& config
    )
    {
        try
        {
            BufferPtr buf = std::make_shared<Buffer_public_ctor>(
                device,
                config
            );

            VkBufferCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = buf->config().flags,
                .size = buf->config().size,
                .usage = buf->config().usage,
                .sharingMode = buf->config().sharing_mode,

                .queueFamilyIndexCount =
                (uint32_t)buf->config().queue_family_indices.size(),

                .pQueueFamilyIndices = buf->config().queue_family_indices.data()
            };

            VkResult vk_result = vkCreateBuffer(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &buf->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }

            VkMemoryRequirements vk_mem_requirements;
            vkGetBufferMemoryRequirements(
                device->handle(),
                buf->handle(),
                &vk_mem_requirements
            );
            buf->_memory_requirements = MemoryRequirements_from_vk(
                vk_mem_requirements
            );

            return buf;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create buffer: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void Buffer::bind_memory(
        const DeviceMemoryPtr& memory,
        VkDeviceSize memory_offset
    )
    {
        try
        {
            VkResult vk_result = vkBindBufferMemory(
                lock_wptr(device())->handle(),
                handle(),
                memory->handle(),
                memory_offset
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to bind buffer memory: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    Buffer::~Buffer()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyBuffer(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    Buffer::Buffer(const DevicePtr& device, const BufferConfig& config)
        : _device(device), _config(config)
    {}

    DeviceMemoryPtr DeviceMemory::allocate(
        const DevicePtr& device,
        const DeviceMemoryConfig& config
    )
    {
        try
        {
            DeviceMemoryPtr mem = std::make_shared<DeviceMemory_public_ctor>(
                device,
                config
            );

            VkMemoryAllocateInfo allocate_info{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .pNext = nullptr,
                .allocationSize = mem->config().allocation_size,
                .memoryTypeIndex = mem->config().memory_type_index
            };

            VkResult vk_result = vkAllocateMemory(
                device->handle(),
                &allocate_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &mem->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return mem;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to allocate device memory: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void* DeviceMemory::map(VkDeviceSize offset, VkDeviceSize size)
    {
        try
        {
            void* p;
            VkResult vk_result = vkMapMemory(
                lock_wptr(device())->handle(),
                handle(),
                offset,
                size,
                0,
                &p
            );
            if (vk_result != VK_SUCCESS)
            {
                unmap();
                throw Error(vk_result);
            }
            return p;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to map device memory: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void DeviceMemory::unmap()
    {
        if (device().expired())
        {
            return;
        }
        vkUnmapMemory(device().lock()->handle(), handle());
    }

    void DeviceMemory::flush_mapped_range(
        VkDeviceSize offset,
        VkDeviceSize size
    )
    {
        try
        {
            VkMappedMemoryRange vk_mapped_range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = handle(),
                .offset = offset,
                .size = size
            };

            VkResult vk_result = vkFlushMappedMemoryRanges(
                lock_wptr(device())->handle(),
                1,
                &vk_mapped_range
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to flush mapped device memory range: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void DeviceMemory::invalidate_mapped_range(
        VkDeviceSize offset,
        VkDeviceSize size
    )
    {
        try
        {
            VkMappedMemoryRange vk_mapped_range{
                .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                .pNext = nullptr,
                .memory = handle(),
                .offset = offset,
                .size = size
            };

            VkResult vk_result = vkInvalidateMappedMemoryRanges(
                lock_wptr(device())->handle(),
                1,
                &vk_mapped_range
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to invalidate mapped device memory range: "
                + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    void DeviceMemory::upload(void* data, VkDeviceSize data_size)
    {
        try
        {
            if (data_size > config().allocation_size)
            {
                throw Error("data is too big");
            }

            void* mapped = map(0, config().allocation_size);
            std::copy(
                (uint8_t*)data,
                (uint8_t*)data + data_size,
                (uint8_t*)mapped
            );
            flush_mapped_range(0, VK_WHOLE_SIZE);
            unmap();
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to upload data to device memory: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    DeviceMemory::~DeviceMemory()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkFreeMemory(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    DeviceMemory::DeviceMemory(
        const DevicePtr& device,
        const DeviceMemoryConfig& config
    )
        : _device(device), _config(config)
    {}

    void DescriptorSet::update_sets(
        const DevicePtr& device,
        const std::vector<WriteDescriptorSet>& writes,
        const std::vector<CopyDescriptorSet>& copies
    )
    {
        if (writes.empty() && copies.empty())
        {
            return;
        }

        try
        {
            std::vector<VkWriteDescriptorSet> vk_writes(writes.size());
            std::vector<std::vector<VkDescriptorImageInfo>>
                wastes_vk_image_infos(writes.size());
            std::vector<std::vector<VkDescriptorBufferInfo>>
                wastes_vk_buffer_infos(writes.size());
            std::vector<std::vector<VkBufferView>>
                wastes_vk_texel_buffer_views(writes.size());
            for (size_t i = 0; i < writes.size(); i++)
            {
                vk_writes[i] = WriteDescriptorSet_to_vk(
                    writes[i],
                    wastes_vk_image_infos[i],
                    wastes_vk_buffer_infos[i],
                    wastes_vk_texel_buffer_views[i]
                );
            }

            std::vector<VkCopyDescriptorSet> vk_copies(copies.size());
            for (size_t i = 0; i < copies.size(); i++)
            {
                vk_copies[i] = CopyDescriptorSet_to_vk(copies[i]);
            }

            vkUpdateDescriptorSets(
                device->handle(),
                (uint32_t)vk_writes.size(),
                vk_writes.data(),
                (uint32_t)vk_copies.size(),
                vk_copies.data()
            );
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to update descriptor sets: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    DescriptorSet::~DescriptorSet()
    {
        _BV_LOCK_WPTR_OR_RETURN(pool(), pool_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            pool_locked->device(),
            device_locked
        );

        vkFreeDescriptorSets(
            device_locked->handle(),
            pool_locked->handle(),
            1,
            &_handle
        );
    }

    DescriptorSet::DescriptorSet(
        const DescriptorPoolPtr& pool,
        VkDescriptorSet handle
    )
        : _pool(pool), _handle(handle)
    {}

    DescriptorPoolPtr DescriptorPool::create(
        const DevicePtr& device,
        const DescriptorPoolConfig& config
    )
    {
        try
        {
            DescriptorPoolPtr pool =
                std::make_shared<DescriptorPool_public_ctor>(
                    device,
                    config
                );

            std::vector<VkDescriptorPoolSize> vk_pool_sizes(
                pool->config().pool_sizes.size()
            );
            for (size_t i = 0; i < pool->config().pool_sizes.size(); i++)
            {
                vk_pool_sizes[i] = DescriptorPoolSize_to_vk(
                    pool->config().pool_sizes[i]
                );
            }

            VkDescriptorPoolCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = pool->config().flags,
                .maxSets = pool->config().max_sets,
                .poolSizeCount = (uint32_t)vk_pool_sizes.size(),
                .pPoolSizes = vk_pool_sizes.data()
            };

            VkResult vk_result = vkCreateDescriptorPool(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &pool->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return pool;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create descriptor pool: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    DescriptorSetPtr DescriptorPool::allocate_set(
        const DescriptorPoolPtr& pool,
        const DescriptorSetLayoutPtr& set_layout
    )
    {
        try
        {
            VkDescriptorSetLayout vk_set_layout = set_layout->handle();

            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = pool->handle(),
                .descriptorSetCount = 1,
                .pSetLayouts = &vk_set_layout
            };

            VkDescriptorSet vk_set;
            VkResult vk_result = vkAllocateDescriptorSets(
                lock_wptr(pool->device())->handle(),
                &alloc_info,
                &vk_set
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return std::make_shared<DescriptorSet_public_ctor>(
                pool,
                vk_set
            );
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to allocate descriptor set: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::vector<DescriptorSetPtr> DescriptorPool::allocate_sets(
        const DescriptorPoolPtr& pool,
        uint32_t count,
        const std::vector<DescriptorSetLayoutPtr>& set_layouts
    )
    {
        try
        {
            if (count != set_layouts.size())
            {
                throw Error(
                    "there should be the same number of descriptor set layouts "
                    "as the number of sets to create"
                );
            }

            if (count < 1)
            {
                return {};
            }

            std::vector<VkDescriptorSetLayout>vk_set_layouts(
                set_layouts.size()
            );
            for (size_t i = 0; i < set_layouts.size(); i++)
            {
                vk_set_layouts[i] = set_layouts[i]->handle();
            }

            VkDescriptorSetAllocateInfo alloc_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = pool->handle(),
                .descriptorSetCount = count,
                .pSetLayouts = vk_set_layouts.data()
            };

            std::vector<VkDescriptorSet> vk_sets(count);
            VkResult vk_result = vkAllocateDescriptorSets(
                lock_wptr(pool->device())->handle(),
                &alloc_info,
                vk_sets.data()
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }

            std::vector<DescriptorSetPtr> sets(count);
            for (size_t i = 0; i < count; i++)
            {
                sets[i] = std::make_shared<DescriptorSet_public_ctor>(
                    pool,
                    vk_sets[i]
                );
            }
            return sets;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to allocate descriptor set(s): " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    DescriptorPool::~DescriptorPool()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyDescriptorPool(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    DescriptorPool::DescriptorPool(
        const DevicePtr& device,
        const DescriptorPoolConfig& config
    )
        : _device(device), _config(config)
    {}

    BufferViewPtr BufferView::create(
        const DevicePtr& device,
        const BufferPtr& buffer,
        const BufferViewConfig& config
    )
    {
        try
        {
            BufferViewPtr view = std::make_shared<BufferView_public_ctor>(
                device,
                buffer,
                config
            );

            VkBufferViewCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .buffer = lock_wptr(view->buffer())->handle(),
                .format = view->config().format,
                .offset = view->config().offset,
                .range = view->config().range
            };

            VkResult vk_result = vkCreateBufferView(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &view->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return view;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create buffer view: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    BufferView::~BufferView()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyBufferView(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    BufferView::BufferView(
        const DevicePtr& device,
        const BufferPtr& buffer,
        const BufferViewConfig& config
    )
        : _device(device),
        _buffer(buffer),
        _config(config)
    {}

    PipelineCachePtr PipelineCache::create(
        const DevicePtr& device,
        VkPipelineCacheCreateFlags flags,
        const std::vector<uint8_t>& initial_data
    )
    {
        try
        {
            PipelineCachePtr cache =
                std::make_shared<PipelineCache_public_ctor>(
                    device,
                    flags
                );

            VkPipelineCacheCreateInfo create_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                .pNext = nullptr,
                .flags = cache->flags(),
                .initialDataSize = initial_data.size(),
                .pInitialData = initial_data.data()
            };

            VkResult vk_result = vkCreatePipelineCache(
                device->handle(),
                &create_info,
                lock_wptr(device->context())->vk_allocator_ptr(),
                &cache->_handle
            );
            if (vk_result != VK_SUCCESS)
            {
                throw Error(vk_result);
            }
            return cache;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to create pipeline cache: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::vector<uint8_t> PipelineCache::get_cache_data()
    {
        try
        {
            auto device_locked = lock_wptr(device());

            size_t size;
            vkGetPipelineCacheData(
                device_locked->handle(),
                handle(),
                &size,
                nullptr
            );

            std::vector<uint8_t> data(size);
            VkResult vk_result = vkGetPipelineCacheData(
                device_locked->handle(),
                handle(),
                &size,
                data.data()
            );
            if (vk_result != VK_SUCCESS && vk_result != VK_INCOMPLETE)
            {
                throw Error(vk_result);
            }

            return data;
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to get pipeline cache data: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    PipelineCache::~PipelineCache()
    {
        _BV_LOCK_WPTR_OR_RETURN(device(), device_locked);
        _BV_LOCK_WPTR_OR_RETURN(
            device_locked->context(),
            context_locked
        );

        vkDestroyPipelineCache(
            device_locked->handle(),
            handle(),
            context_locked->vk_allocator_ptr()
        );
    }

    PipelineCache::PipelineCache(
        const DevicePtr& device,
        VkPipelineCacheCreateFlags flags
    )
        : _device(device), _flags(flags)
    {}

#pragma endregion

#pragma region helper functions

    std::string cstr_to_std(const char* cstr)
    {
        return (cstr == nullptr) ? std::string() : std::string(cstr);
    }

    bool format_has_depth_component(VkFormat format)
    {
        return
            format == VK_FORMAT_D16_UNORM
            || format == VK_FORMAT_X8_D24_UNORM_PACK32
            || format == VK_FORMAT_D32_SFLOAT
            || format == VK_FORMAT_D16_UNORM_S8_UINT
            || format == VK_FORMAT_D24_UNORM_S8_UINT
            || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    }

    bool format_has_stencil_component(VkFormat format)
    {
        return
            format == VK_FORMAT_S8_UINT
            || format == VK_FORMAT_D16_UNORM_S8_UINT
            || format == VK_FORMAT_D24_UNORM_S8_UINT
            || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    }

#pragma endregion

#pragma region memory management

    static uint32_t find_memory_type_idx(
        const bv::PhysicalDevice& physical_device,
        uint32_t supported_type_bits,
        VkMemoryPropertyFlags required_properties,
        VkDeviceSize required_size
    )
    {
        const auto& mem_props = physical_device.memory_properties();
        for (uint32_t i = 0; i < mem_props.memory_types.size(); i++)
        {
            bool has_required_properties =
                (required_properties & mem_props.memory_types[i].property_flags)
                == required_properties;

            if (!(supported_type_bits & (1 << i)) || !has_required_properties)
            {
                continue;
            }

            VkDeviceSize heap_size = mem_props.memory_heaps[
                mem_props.memory_types[i].heap_index
            ].size;
            if (required_size > heap_size)
            {
                continue;
            }

            return i;
        }
        throw Error("failed to find a suitable memory type");
    }

    MemoryRegion::MemoryRegion(const bv::DeviceMemoryPtr& mem)
        : mem(mem)
    {}

    const bv::DeviceMemoryPtr& MemoryChunk::memory() const
    {
        return region->mem;
    }

    void MemoryChunk::bind(bv::BufferPtr& buffer)
    {
        buffer->bind_memory(memory(), offset());
    }

    void MemoryChunk::bind(bv::ImagePtr& image)
    {
        image->bind_memory(memory(), offset());
    }

    void* MemoryChunk::mapped()
    {
        if (region->mapped == nullptr)
        {
            throw Error("failed to map memory chunk: unmappable memory");
        }
        return (void*)((uint8_t*)region->mapped + offset());
    }

    void MemoryChunk::flush()
    {
        if (region->mapped != nullptr)
        {
            region->mem->flush_mapped_range(0, VK_WHOLE_SIZE);
        }
    }

    MemoryChunk::~MemoryChunk()
    {
        std::scoped_lock lock(*mutex);

        // set the block bits to free
        uint64_t start_block_idx = offset() / block_size;
        uint64_t end_block_idx = _BV_IDIV_CEIL(offset() + size(), block_size);
        for (size_t i = start_block_idx; i < end_block_idx; i++)
        {
            region->blocks[i] = false;
        }
    }

    MemoryChunk::MemoryChunk(
        const std::shared_ptr<std::mutex>& mutex,
        const MemoryRegionPtr& region,
        VkDeviceSize offset,
        VkDeviceSize size,
        VkDeviceSize block_size
    )
        : mutex(mutex),
        region(region),
        _offset(offset),
        _size(size),
        block_size(block_size)
    {}

    MemoryBankPtr MemoryBank::create(
        const DevicePtr& device,
        VkDeviceSize block_size,
        VkDeviceSize min_region_size
    )
    {
        return std::make_shared<MemoryBank_public_ctor>(
            device,
            block_size,
            min_region_size
        );
    }

    MemoryChunkPtr MemoryBank::allocate(
        const bv::MemoryRequirements& requirements,
        VkMemoryPropertyFlags required_properties
    )
    {
        try
        {
            std::scoped_lock lock(*mutex);

            // make sure the chunk size is divisible by the block size
            uint64_t chunk_size = requirements.size;
            if (chunk_size % block_size() != 0)
            {
                chunk_size += block_size() - (chunk_size % block_size());
            }

            uint64_t n_blocks_in_chunk =
                _BV_IDIV_CEIL(chunk_size, block_size());

            const auto& mem_props =
                device()->physical_device().memory_properties();

            for (auto& region : regions)
            {
                // check if region is large enough
                if (chunk_size > region->mem->config().allocation_size)
                {
                    continue;
                }

                // check if region has a compatible memory type
                if (!(
                    requirements.memory_type_bits
                    & (1 << region->mem->config().memory_type_index)
                    ))
                {
                    continue;
                }

                // check if region's memory type has the required properties
                uint32_t mem_type_idx = region->mem->config().memory_type_index;
                bool has_required_properties =
                    (required_properties
                        & mem_props.memory_types[mem_type_idx].property_flags)
                    == required_properties;
                if (!has_required_properties)
                {
                    continue;
                }

                // try to find a free range and return a chunk if found
                uint64_t start_block_idx = 0;
                while (true)
                {
                    while (start_block_idx < region->blocks.size()
                        && region->blocks[start_block_idx])
                    {
                        start_block_idx++;
                    }
                    if (start_block_idx >= region->blocks.size())
                    {
                        break;
                    }

                    uint64_t end_block_idx = start_block_idx + 1;
                    while (end_block_idx < region->blocks.size()
                        && !region->blocks[end_block_idx])
                    {
                        end_block_idx++;
                    }

                    uint64_t n_free_blocks = end_block_idx - start_block_idx;

                    // if we have found a free and usable block range
                    if (n_free_blocks >= n_blocks_in_chunk)
                    {
                        // figure out the byte offset in the region
                        VkDeviceSize offs = start_block_idx * block_size();

                        // adjust the offset if it doesn't meet the alignment
                        // requirements
                        if (offs % requirements.alignment != 0)
                        {
                            offs +=
                                requirements.alignment
                                - (offs % requirements.alignment);
                        }

                        // recalculate the start block index
                        start_block_idx = offs / block_size();

                        // if the start block index is now stepping into the
                        // end of the range, we'll restart the search.
                        if (start_block_idx >= end_block_idx)
                        {
                            continue;
                        }

                        // recalculate the new number of free blocks
                        n_free_blocks = end_block_idx - start_block_idx;

                        // if the new number of free blocks is no longer large
                        // enough, we'll restart the search.
                        if (n_free_blocks < n_blocks_in_chunk)
                        {
                            start_block_idx = end_block_idx;
                            continue;
                        }

                        // set the block bits to allocated
                        for (size_t i = start_block_idx;
                            i < start_block_idx + n_blocks_in_chunk;
                            i++)
                        {
                            region->blocks[i] = true;
                        }

                        // make a copy of the region pointer to avoid losing it
                        // when we delete empty regions.
                        bv::MemoryRegionPtr region_ptr_copy = region;

                        // delete empty regions
                        delete_empty_regions();

                        // return a new chunk based on the region
                        return std::make_shared<MemoryChunk_public_ctor>(
                            mutex,
                            region_ptr_copy,
                            offs,
                            chunk_size,
                            block_size()
                        );
                    }

                    // continue the search
                    start_block_idx = end_block_idx;
                }
            }

            // couldn't find a usable range in any of the regions, so we'll
            // create a new region and use it instead.

            // figure out the allocation size and make sure it's divisible by
            // the block size.
            VkDeviceSize region_size = std::max(chunk_size, min_region_size());
            if (region_size % block_size() != 0)
            {
                region_size += block_size() - (region_size % block_size());
            }

            // find suitable memory type index
            uint32_t memory_type_idx = find_memory_type_idx(
                device()->physical_device(),
                requirements.memory_type_bits,
                required_properties,
                region_size
            );

            // allocate memory
            auto mem = bv::DeviceMemory::allocate(
                device(),
                {
                    .allocation_size = region_size,
                    .memory_type_index = memory_type_idx
                }
            );

            // create a region based on that memory
            auto new_region = std::make_shared<MemoryRegion_public_ctor>(mem);

            // setup the block bitset for the region
            new_region->blocks.resize(
                _BV_IDIV_CEIL(region_size, block_size()),
                false
            );
            for (size_t i = 0; i < n_blocks_in_chunk; i++)
            {
                new_region->blocks[i] = true;
            }

            // map the memory if it's mappable
            bool is_mappable =
                (required_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                || (required_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (is_mappable)
            {
                new_region->mapped = mem->map(0, VK_WHOLE_SIZE);
            }

            // delete empty regions
            delete_empty_regions();

            // add the region to our regions vector
            regions.push_back(new_region);

            // return a new chunk based on the region
            return std::make_shared<MemoryChunk_public_ctor>(
                mutex,
                new_region,
                0,
                chunk_size,
                block_size()
            );
        }
        catch (const Error& e)
        {
            throw Error(
                "failed to allocate chunk from memory bank: " + e.to_string(),
                e.vk_result(),
                true
            );
        }
    }

    std::string MemoryBank::to_string()
    {
        std::scoped_lock lock(*mutex);

        std::string s = fmt::format(
            "-----------------------------------------\n"
            "memory bank status\n"
            "  n. regions: {}\n"
            "  block size: {}\n",
            regions.size(),
            block_size()
        );
        for (size_t i = 0; i < regions.size(); i++)
        {
            const auto& region = regions[i];
            s += fmt::format(
                "-----------------------------------------\n"
                "region {}\n"
                "  size: {}\n"
                "  mapped: {}\n"
                "  blocks: {} blocks allocated out of {}\n",
                i,
                region->mem->config().allocation_size,
                region->mapped != nullptr,
                region->blocks.count(),
                region->blocks.size()
            );
        }
        s += "-----------------------------------------\n";
        return s;
    }

    MemoryBank::~MemoryBank()
    {
        std::scoped_lock lock(*mutex);
    }

    MemoryBank::MemoryBank(
        const bv::DevicePtr& device,
        VkDeviceSize block_size,
        VkDeviceSize min_region_size
    )
        : _device(device),
        mutex(std::make_shared<std::mutex>()),
        _block_size(block_size),
        _min_region_size(min_region_size)
    {}

    void MemoryBank::delete_empty_regions()
    {
        for (size_t i = 0; i < regions.size();)
        {
            if (regions[i]->blocks.none())
            {
                regions.erase(regions.begin() + i);
            }
            else
            {
                i++;
            }
        }

    }

#pragma endregion

#pragma region Vulkan callbacks

    static void* vk_allocation_callback(
        void* p_user_data,
        size_t size,
        size_t alignment,
        VkSystemAllocationScope allocation_scope
    )
    {
        if (!p_user_data)
        {
            throw Error("Vulkan allocation callback called with no user data");
        }

        Allocator* allocator = (Allocator*)p_user_data;
        return allocator->allocate(
            size,
            alignment,
            allocation_scope
        );
    }

    static void* vk_reallocation_callback(
        void* p_user_data,
        void* p_original,
        size_t size,
        size_t alignment,
        VkSystemAllocationScope allocation_scope
    )
    {
        if (!p_user_data)
        {
            throw Error(
                "Vulkan reallocation callback called with no user data"
            );
        }

        Allocator* allocator = (Allocator*)p_user_data;
        return allocator->reallocate(
            p_original,
            size,
            alignment,
            allocation_scope
        );
    }

    static void vk_free_callback(
        void* p_user_data,
        void* p_memory
    )
    {
        if (!p_user_data)
        {
            throw Error(
                "Vulkan free callback called with no user data"
            );
        }

        Allocator* allocator = (Allocator*)p_user_data;
        allocator->free(p_memory);
    }

    static void vk_internal_allocation_notification(
        void* p_user_data,
        size_t size,
        VkInternalAllocationType allocation_type,
        VkSystemAllocationScope allocation_scope
    )
    {
        if (!p_user_data)
        {
            throw Error(
                "Vulkan internal allocation notification called with no user "
                "data"
            );
        }

        Allocator* allocator = (Allocator*)p_user_data;
        allocator->internal_allocation_notification(
            size,
            allocation_type,
            allocation_scope
        );
    }

    static void vk_internal_free_notification(
        void* p_user_data,
        size_t size,
        VkInternalAllocationType allocation_type,
        VkSystemAllocationScope allocation_scope
    )
    {
        if (!p_user_data)
        {
            throw Error(
                "Vulkan internal free notification called with no user data"
            );
        }

        Allocator* allocator = (Allocator*)p_user_data;
        allocator->internal_free_notification(
            size,
            allocation_type,
            allocation_scope
        );
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL vk_debug_callback(
        VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
        VkDebugUtilsMessageTypeFlagsEXT message_types,
        const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
        void* p_user_data
    )
    {
        if (!p_user_data)
        {
            throw Error("Vulkan debug callback called with no user data");
        }

        DebugMessenger* messenger = (DebugMessenger*)p_user_data;
        messenger->callback()(
            message_severity,
            message_types,
            DebugMessageData_from_vk(*p_callback_data)
            );

        return VK_FALSE;
    }

#pragma endregion

}
