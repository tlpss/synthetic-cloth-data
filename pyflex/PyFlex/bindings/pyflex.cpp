#include <bindings/main.cpp>
#include <bindings/customizedAPI.h>

char rope_path[100];
char box_high_path[100];
char sphere_path[100];

char *make_path(char *full_path, std::string path)
{
    strcpy(full_path, getenv("PYFLEXROOT"));
    strcat(full_path, path.c_str());
    return full_path;
}

void pyflex_init(bool headless = false, bool render = true, int camera_width = 720, int camera_height = 720, int msaaSamples = 8)
{
    g_screenWidth = camera_width;
    g_screenHeight = camera_height;
    g_msaaSamples = msaaSamples;

    g_headless = headless;
    g_render = render;
    if (g_headless)
    {
        g_interop = false;
        g_pause = false;
    }

    g_scenes.push_back(new EmptyScene("Empty Scene"));
    // g_scenes.push_back(new SoftgymCloth("Softgym Flag Cloth"));
    // g_scenes.push_back(new GranularScene("Granular Scene"));

    switch (g_graphics)
    {
    case 0:
        break;
    case 1:
        break;
    case 2:
        // workaround for a driver issue with D3D12 with msaa, force it to off
        // options.numMsaaSamples = 1;
        // Currently interop doesn't work on d3d12
        g_interop = false;
        break;
    default:
        assert(0);
    }

    // Create the demo context
    CreateDemoContext(g_graphics);

    std::string str;
    str = "Flex Demo (Compute: CUDA) ";

    switch (g_graphics)
    {
    case 0:
        str += "(Graphics: OpenGL)";
        break;
    case 1:
        str += "(Graphics: DX11)";
        break;
    case 2:
        str += "(Graphics: DX12)";
        break;
    }
    const char *title = str.c_str();

    if (!g_headless)
    {
        SDLInit(title);

        // init graphics
        RenderInitOptions options;
        options.window = g_window;
        options.numMsaaSamples = g_msaaSamples;
        options.asyncComputeBenchmark = g_asyncComputeBenchmark;
        options.defaultFontHeight = -1;
        options.fullscreen = g_fullscreen;

        InitRender(options);

        ReshapeWindow(g_screenWidth, g_screenHeight);
    }
    else if (g_render == true)
    {
        RenderInitOptions options;
        options.numMsaaSamples = g_msaaSamples;

        InitRenderHeadless(options, g_screenWidth, g_screenHeight);
        g_fluidRenderer = CreateFluidRenderer(g_screenWidth, g_screenHeight);
    }

    NvFlexInitDesc desc;
    desc.deviceIndex = g_device;
    desc.enableExtensions = g_extensions;
    desc.renderDevice = 0;
    desc.renderContext = 0;
    desc.computeContext = 0;
    desc.computeType = eNvFlexCUDA;

    // Init Flex library, note that no CUDA methods should be called before this
    // point to ensure we get the device context we want
    g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);

    if (g_Error || g_flexLib == nullptr)
    {
        printf("Could not initialize Flex, exiting.\n");
        exit(-1);
    }

    // store device name
    strcpy(g_deviceName, NvFlexGetDeviceName(g_flexLib));
    printf("Compute Device: %s\n\n", g_deviceName);

    if (g_benchmark)
        g_scene = BenchmarkInit();

    // create shadow maps
    if (g_render)
    {
        g_shadowMap = ShadowCreate();
    }

    // init default scene
    printf("Pyflex init done!\n");
}

void pyflex_clean()
{
    if (g_fluidRenderer)
        DestroyFluidRenderer(g_fluidRenderer);
    if (g_fluidRenderBuffers)
        DestroyFluidRenderBuffers(g_fluidRenderBuffers);
    if (g_diffuseRenderBuffers)
        DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);
    if (g_shadowMap)
        ShadowDestroy(g_shadowMap);
    Shutdown();
    if (g_headless == false)
    {
        DestroyRender();
        if (g_window)
            SDL_DestroyWindow(g_window);
        SDL_Quit();
    }
}

int main()
{
    pyflex_init();
    pyflex_clean();

    return 0;
}

void SDL_EventFunc()
{
    SDL_Event e;
    while (SDL_PollEvent(&e))
    {
        switch (e.type)
        {
        case SDL_QUIT:
            break;

        case SDL_KEYDOWN:
            InputArrowKeysDown(e.key.keysym.sym, 0, 0);
            InputKeyboardDown(e.key.keysym.sym, 0, 0);
            break;

        case SDL_KEYUP:
            if (e.key.keysym.sym < 256 && (e.key.keysym.mod == 0 || (e.key.keysym.mod & KMOD_NUM)))
                InputKeyboardUp(e.key.keysym.sym, 0, 0);
            InputArrowKeysUp(e.key.keysym.sym, 0, 0);
            break;

        case SDL_MOUSEMOTION:
            if (e.motion.state)
                MouseMotionFunc(e.motion.state, e.motion.x, e.motion.y);
            else
                MousePassiveMotionFunc(e.motion.x, e.motion.y);
            break;

        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
            MouseFunc(e.button.button, e.button.state, e.motion.x, e.motion.y);
            break;

        case SDL_WINDOWEVENT:
            if (e.window.windowID == g_windowId)
            {
                if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED)
                    ReshapeWindow(e.window.data1, e.window.data2);
            }
            break;

        case SDL_WINDOWEVENT_LEAVE:
            g_camVel = Vec3(0.0f, 0.0f, 0.0f);
            break;

        case SDL_CONTROLLERBUTTONUP:
        case SDL_CONTROLLERBUTTONDOWN:
            ControllerButtonEvent(e.cbutton);
            break;

        case SDL_JOYDEVICEADDED:
        case SDL_JOYDEVICEREMOVED:
            ControllerDeviceUpdate();
            break;
        }
    }
}

void pyflex_step(py::array_t<float> update_params, int capture, char *path)
{
    UpdateFrame(update_params);
    SDL_EventFunc();
}

float rand_float(float LO, float HI)
{
    return LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
}

void pyflex_set_scene(int scene_idx, py::dict scene_params)
{
    g_scene = scene_idx;
    g_selectedScene = g_scene;
    Init(g_selectedScene, scene_params);
}

void pyflex_MapShapeBuffers(SimBuffers *buffers)
{
    buffers->shapeGeometry.map();
    buffers->shapePositions.map();
    buffers->shapeRotations.map();
    buffers->shapePrevPositions.map();
    buffers->shapePrevRotations.map();
    buffers->shapeFlags.map();
}

void pyflex_UnmapShapeBuffers(SimBuffers *buffers)
{
    buffers->shapeGeometry.unmap();
    buffers->shapePositions.unmap();
    buffers->shapeRotations.unmap();
    buffers->shapePrevPositions.unmap();
    buffers->shapePrevRotations.unmap();
    buffers->shapeFlags.unmap();
}

void pyflex_add_capsule(py::array_t<float> params, py::array_t<float> lower_pos, py::array_t<float> quat_)
{
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_params = (float *)params.request().ptr;
    float capsule_radius = ptr_params[0];
    float halfheight = ptr_params[1];

    auto ptr_lower_pos = (float *)lower_pos.request().ptr;
    Vec3 lower_position = Vec3(ptr_lower_pos[0], ptr_lower_pos[1], ptr_lower_pos[2]);

    auto ptr_quat = (float *)quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddCapsule(capsule_radius, halfheight, lower_position, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}

void pyflex_add_box(py::array_t<float> halfEdge_, py::array_t<float> center_, py::array_t<float> quat_, int trigger)
{
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_halfEdge = (float *)halfEdge_.request().ptr;
    Vec3 halfEdge = Vec3(ptr_halfEdge[0], ptr_halfEdge[1], ptr_halfEdge[2]);

    auto ptr_center = (float *)center_.request().ptr;
    Vec3 center = Vec3(ptr_center[0], ptr_center[1], ptr_center[2]);

    auto ptr_quat = (float *)quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    // cout << "trigger is " << trigger << endl;
    AddBox(halfEdge, center, quat, trigger);

    pyflex_UnmapShapeBuffers(g_buffers);
}

void pyflex_pop_shape(int num)
{
    pyflex_MapShapeBuffers(g_buffers);
    PopShape(num);
    pyflex_UnmapShapeBuffers(g_buffers);
}

void pyflex_add_sphere(float radius, py::array_t<float> position_, py::array_t<float> quat_)
{
    pyflex_MapShapeBuffers(g_buffers);

    auto ptr_center = (float *)position_.request().ptr;
    Vec3 center = Vec3(ptr_center[0], ptr_center[1], ptr_center[2]);

    auto ptr_quat = (float *)quat_.request().ptr;
    Quat quat = Quat(ptr_quat[0], ptr_quat[1], ptr_quat[2], ptr_quat[3]);

    AddSphere(radius, center, quat);

    pyflex_UnmapShapeBuffers(g_buffers);
}

int pyflex_get_n_particles()
{
    g_buffers->positions.map();
    int n_particles = g_buffers->positions.size();
    g_buffers->positions.unmap();
    return n_particles;
}

int pyflex_get_n_shapes()
{
    g_buffers->shapePositions.map();
    int n_shapes = g_buffers->shapePositions.size();
    g_buffers->shapePositions.unmap();
    return n_shapes;
}

py::array_t<int> pyflex_get_groups()
{
    g_buffers->phases.map();

    auto groups = py::array_t<int>((size_t)g_buffers->phases.size());
    auto ptr = (int *)groups.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->phases.size(); i++)
    {
        ptr[i] = g_buffers->phases[i] & 0xfffff; // Flex 1.1 manual actually says 24 bits while it is actually 20 bits
    }

    g_buffers->phases.unmap();

    return groups;
}

void pyflex_set_groups(py::array_t<int> groups)
{
    //    if (not set_color)
    //        cout<<"Warning: Overloading GroupMask for colors. Make sure the eFlexPhaseSelfCollide is set!"<<endl;
    g_buffers->phases.map();

    auto buf = groups.request();
    auto ptr = (int *)buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->phases.size(); i++)
    {
        g_buffers->phases[i] = (g_buffers->phases[i] & ~0xfffff) | (ptr[i] & 0xfffff);
    }

    g_buffers->phases.unmap();

    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
}

py::array_t<int> pyflex_get_phases()
{
    g_buffers->phases.map();

    auto phases = py::array_t<int>((size_t)g_buffers->phases.size());
    auto ptr = (int *)phases.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->phases.size(); i++)
    {
        ptr[i] = g_buffers->phases[i];
    }

    g_buffers->phases.unmap();

    return phases;
}

void pyflex_set_phases(py::array_t<int> phases)
{
    //    if (not set_color)
    //        cout<<"Warning: Overloading GroupMask for colors. Make sure the eFlexPhaseSelfCollide is set!"<<endl;
    g_buffers->phases.map();

    auto buf = phases.request();
    auto ptr = (int *)buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->phases.size(); i++)
    {
        g_buffers->phases[i] = ptr[i];
    }

    g_buffers->phases.unmap();

    NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
}

py::array_t<float> pyflex_get_positions()
{
    g_buffers->positions.map();
    auto positions = py::array_t<float>((size_t)g_buffers->positions.size() * 4);
    auto ptr = (float *)positions.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->positions.size(); i++)
    {
        ptr[i * 4] = g_buffers->positions[i].x;
        ptr[i * 4 + 1] = g_buffers->positions[i].y;
        ptr[i * 4 + 2] = g_buffers->positions[i].z;
        ptr[i * 4 + 3] = g_buffers->positions[i].w;
    }

    g_buffers->positions.unmap();

    return positions;
}

py::array_t<int> pyflex_get_edges()
{
    g_buffers->springIndices.map();
    auto edges = py::array_t<int>((size_t)g_buffers->springIndices.size());
    auto ptr = (int *)edges.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->springIndices.size(); i++)
    {
        ptr[i] = g_buffers->springIndices[i];
    }

    g_buffers->springIndices.unmap();

    return edges;
}

py::array_t<int> pyflex_get_faces()
{
    g_buffers->triangles.map();
    auto triangles = py::array_t<int>((size_t)g_buffers->triangles.size());
    auto ptr = (int *)triangles.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->triangles.size(); i++)
    {
        ptr[i] = g_buffers->triangles[i];
    }

    g_buffers->triangles.unmap();
    return triangles;
}

void pyflex_set_positions(py::array_t<float> positions)
{
    g_buffers->positions.map();

    auto buf = positions.request();
    auto ptr = (float *)buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->positions.size(); i++)
    {
        g_buffers->positions[i].x = ptr[i * 4];
        g_buffers->positions[i].y = ptr[i * 4 + 1];
        g_buffers->positions[i].z = ptr[i * 4 + 2];
        g_buffers->positions[i].w = ptr[i * 4 + 3];
    }

    g_buffers->positions.unmap();

    NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
}

void pyflex_add_rigid_body(py::array_t<float> positions, py::array_t<float> velocities, int num, py::array_t<float> lower)
{
    auto bufp = positions.request();
    auto position_ptr = (float *)bufp.ptr;

    auto bufv = velocities.request();
    auto velocity_ptr = (float *)bufv.ptr;

    auto bufl = lower.request();
    auto lower_ptr = (float *)bufl.ptr;

    MapBuffers(g_buffers);

    // if (g_buffers->rigidIndices.empty())
    // 	g_buffers->rigidOffsets.push_back(0);

    int phase = NvFlexMakePhase(5, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
    for (size_t i = 0; i < (size_t)num; i++)
    {
        g_buffers->activeIndices.push_back(int(g_buffers->activeIndices.size()));
        // g_buffers->rigidIndices.push_back(int(g_buffers->positions.size()));
        Vec3 position = Vec3(lower_ptr[0], lower_ptr[1], lower_ptr[2]) + Vec3(position_ptr[i * 4], position_ptr[i * 4 + 1], position_ptr[i * 4 + 2]);
        g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, position_ptr[i * 4 + 3]));
        Vec3 velocity = Vec3(velocity_ptr[i * 3], velocity_ptr[i * 3 + 1], velocity_ptr[i * 3 + 2]);
        g_buffers->velocities.push_back(velocity);
        g_buffers->phases.push_back(phase);
    }

    // g_buffers->rigidCoefficients.push_back(1.0);
    // g_buffers->rigidOffsets.push_back(int(g_buffers->rigidIndices.size()));

    // g_buffers->activeIndices.resize(g_buffers->positions.size());
    // for (int i = 0; i < g_buffers->activeIndices.size(); ++i)
    //     printf("active particle idx: %d %d \n", i, g_buffers->activeIndices[i]);

    // builds rigids constraints
    // if (g_buffers->rigidOffsets.size()) {
    //     assert(g_buffers->rigidOffsets.size() > 1);

    //     const int numRigids = g_buffers->rigidOffsets.size() - 1;

    //     // If the centers of mass for the rigids are not yet computed, this is done here
    //     // (If the CreateParticleShape method is used instead of the NvFlexExt methods, the centers of mass will be calculated here)
    //     if (g_buffers->rigidTranslations.size() == 0) {
    //         g_buffers->rigidTranslations.resize(g_buffers->rigidOffsets.size() - 1, Vec3());
    //         CalculateRigidCentersOfMass(&g_buffers->positions[0], g_buffers->positions.size(), &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids);
    //     }

    //     // calculate local rest space positions
    //     g_buffers->rigidLocalPositions.resize(g_buffers->rigidOffsets.back());
    //     CalculateRigidLocalPositions(&g_buffers->positions[0], &g_buffers->rigidOffsets[0], &g_buffers->rigidTranslations[0], &g_buffers->rigidIndices[0], numRigids, &g_buffers->rigidLocalPositions[0]);

    //     // set rigidRotations to correct length, probably NULL up until here
    //     g_buffers->rigidRotations.resize(g_buffers->rigidOffsets.size() - 1, Quat());
    // }
    uint32_t numParticles = g_buffers->positions.size();

    UnmapBuffers(g_buffers);

    // reset pyflex solvers
    // NvFlexSetParams(g_solver, &g_params);
    // NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
    // NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
    // NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
    // NvFlexSetNormals(g_solver, g_buffers->normals.buffer, nullptr);
    // NvFlexSetRestParticles(g_solver, g_buffers->restPositions.buffer, nullptr);

    NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);
    // printf("ok till here\n");
    NvFlexSetActiveCount(g_solver, numParticles);
    // NvFlexSetRigids(g_solver, g_buffers->rigidOffsets.buffer, g_buffers->rigidIndices.buffer,
    //     g_buffers->rigidLocalPositions.buffer, g_buffers->rigidLocalNormals.buffer,
    //     g_buffers->rigidCoefficients.buffer, g_buffers->rigidPlasticThresholds.buffer,
    //     g_buffers->rigidPlasticCreeps.buffer, g_buffers->rigidRotations.buffer,
    //     g_buffers->rigidTranslations.buffer, g_buffers->rigidOffsets.size() - 1, g_buffers->rigidIndices.size());
    // printf("also ok here\n");
}

py::array_t<float> pyflex_get_restPositions()
{
    g_buffers->restPositions.map();

    auto restPositions = py::array_t<float>((size_t)g_buffers->restPositions.size() * 4);
    auto ptr = (float *)restPositions.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->restPositions.size(); i++)
    {
        ptr[i * 4] = g_buffers->restPositions[i].x;
        ptr[i * 4 + 1] = g_buffers->restPositions[i].y;
        ptr[i * 4 + 2] = g_buffers->restPositions[i].z;
        ptr[i * 4 + 3] = g_buffers->restPositions[i].w;
    }

    g_buffers->restPositions.unmap();

    return restPositions;
}

py::array_t<int> pyflex_get_rigidOffsets()
{
    g_buffers->rigidOffsets.map();

    auto rigidOffsets = py::array_t<int>((size_t)g_buffers->rigidOffsets.size());
    auto ptr = (int *)rigidOffsets.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->rigidOffsets.size(); i++)
    {
        ptr[i] = g_buffers->rigidOffsets[i];
    }

    g_buffers->rigidOffsets.unmap();

    return rigidOffsets;
}

py::array_t<int> pyflex_get_rigidIndices()
{
    g_buffers->rigidIndices.map();

    auto rigidIndices = py::array_t<int>((size_t)g_buffers->rigidIndices.size());
    auto ptr = (int *)rigidIndices.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->rigidIndices.size(); i++)
    {
        ptr[i] = g_buffers->rigidIndices[i];
    }

    g_buffers->rigidIndices.unmap();

    return rigidIndices;
}

int pyflex_get_n_rigidPositions()
{
    g_buffers->rigidLocalPositions.map();
    int n_rigidPositions = g_buffers->rigidLocalPositions.size();
    g_buffers->rigidLocalPositions.unmap();
    return n_rigidPositions;
}

py::array_t<float> pyflex_get_rigidLocalPositions()
{
    g_buffers->rigidLocalPositions.map();

    auto rigidLocalPositions = py::array_t<float>((size_t)g_buffers->rigidLocalPositions.size() * 3);
    auto ptr = (float *)rigidLocalPositions.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->rigidLocalPositions.size(); i++)
    {
        ptr[i * 3] = g_buffers->rigidLocalPositions[i].x;
        ptr[i * 3 + 1] = g_buffers->rigidLocalPositions[i].y;
        ptr[i * 3 + 2] = g_buffers->rigidLocalPositions[i].z;
    }

    g_buffers->rigidLocalPositions.unmap();

    return rigidLocalPositions;
}

py::array_t<float> pyflex_get_rigidGlobalPositions()
{
    g_buffers->rigidOffsets.map();
    g_buffers->rigidIndices.map();
    g_buffers->rigidLocalPositions.map();
    g_buffers->rigidTranslations.map();
    g_buffers->rigidRotations.map();

    auto rigidGlobalPositions = py::array_t<float>((size_t)g_buffers->positions.size() * 3);
    auto ptr = (float *)rigidGlobalPositions.request().ptr;

    int count = 0;
    int numRigids = g_buffers->rigidOffsets.size() - 1;
    float n_clusters[g_buffers->positions.size()] = {0};

    for (int i = 0; i < numRigids; i++)
    {
        const int st = g_buffers->rigidOffsets[i];
        const int ed = g_buffers->rigidOffsets[i + 1];

        assert(ed - st);

        for (int j = st; j < ed; j++)
        {
            const int r = g_buffers->rigidIndices[j];
            Vec3 p = Rotate(g_buffers->rigidRotations[i], g_buffers->rigidLocalPositions[count++]) +
                     g_buffers->rigidTranslations[i];

            if (n_clusters[r] == 0)
            {
                ptr[r * 3] = p.x;
                ptr[r * 3 + 1] = p.y;
                ptr[r * 3 + 2] = p.z;
            }
            else
            {
                ptr[r * 3] += p.x;
                ptr[r * 3 + 1] += p.y;
                ptr[r * 3 + 2] += p.z;
            }
            n_clusters[r] += 1;
        }
    }

    for (int i = 0; i < g_buffers->positions.size(); i++)
    {
        if (n_clusters[i] > 0)
        {
            ptr[i * 3] /= n_clusters[i];
            ptr[i * 3 + 1] /= n_clusters[i];
            ptr[i * 3 + 2] /= n_clusters[i];
        }
    }

    g_buffers->rigidOffsets.unmap();
    g_buffers->rigidIndices.unmap();
    g_buffers->rigidLocalPositions.unmap();
    g_buffers->rigidTranslations.unmap();
    g_buffers->rigidRotations.unmap();

    return rigidGlobalPositions;
}

int pyflex_get_n_rigids()
{
    g_buffers->rigidRotations.map();
    int n_rigids = g_buffers->rigidRotations.size();
    g_buffers->rigidRotations.unmap();
    return n_rigids;
}

py::array_t<float> pyflex_get_rigidRotations()
{
    g_buffers->rigidRotations.map();

    auto rigidRotations = py::array_t<float>((size_t)g_buffers->rigidRotations.size() * 4);
    auto ptr = (float *)rigidRotations.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->rigidRotations.size(); i++)
    {
        ptr[i * 4] = g_buffers->rigidRotations[i].x;
        ptr[i * 4 + 1] = g_buffers->rigidRotations[i].y;
        ptr[i * 4 + 2] = g_buffers->rigidRotations[i].z;
        ptr[i * 4 + 3] = g_buffers->rigidRotations[i].w;
    }

    g_buffers->rigidRotations.unmap();

    return rigidRotations;
}

py::array_t<float> pyflex_get_rigidTranslations()
{
    g_buffers->rigidTranslations.map();

    auto rigidTranslations = py::array_t<float>((size_t)g_buffers->rigidTranslations.size() * 3);
    auto ptr = (float *)rigidTranslations.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->rigidTranslations.size(); i++)
    {
        ptr[i * 3] = g_buffers->rigidTranslations[i].x;
        ptr[i * 3 + 1] = g_buffers->rigidTranslations[i].y;
        ptr[i * 3 + 2] = g_buffers->rigidTranslations[i].z;
    }

    g_buffers->rigidTranslations.unmap();

    return rigidTranslations;
}

py::array_t<float> pyflex_get_velocities()
{
    g_buffers->velocities.map();

    auto velocities = py::array_t<float>((size_t)g_buffers->velocities.size() * 3);
    auto ptr = (float *)velocities.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->velocities.size(); i++)
    {
        ptr[i * 3] = g_buffers->velocities[i].x;
        ptr[i * 3 + 1] = g_buffers->velocities[i].y;
        ptr[i * 3 + 2] = g_buffers->velocities[i].z;
    }

    g_buffers->velocities.unmap();

    return velocities;
}

void pyflex_set_velocities(py::array_t<float> velocities)
{
    g_buffers->velocities.map();

    auto buf = velocities.request();
    auto ptr = (float *)buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->velocities.size(); i++)
    {
        g_buffers->velocities[i].x = ptr[i * 3];
        g_buffers->velocities[i].y = ptr[i * 3 + 1];
        g_buffers->velocities[i].z = ptr[i * 3 + 2];
    }

    g_buffers->velocities.unmap();
}

py::array_t<float> pyflex_get_shape_states()
{
    pyflex_MapShapeBuffers(g_buffers);

    // position + prev_position + rotation + prev_rotation
    auto states = py::array_t<float>((size_t)g_buffers->shapePositions.size() * (3 + 3 + 4 + 4));
    auto buf = states.request();
    auto ptr = (float *)buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->shapePositions.size(); i++)
    {
        ptr[i * 14] = g_buffers->shapePositions[i].x;
        ptr[i * 14 + 1] = g_buffers->shapePositions[i].y;
        ptr[i * 14 + 2] = g_buffers->shapePositions[i].z;

        ptr[i * 14 + 3] = g_buffers->shapePrevPositions[i].x;
        ptr[i * 14 + 4] = g_buffers->shapePrevPositions[i].y;
        ptr[i * 14 + 5] = g_buffers->shapePrevPositions[i].z;

        ptr[i * 14 + 6] = g_buffers->shapeRotations[i].x;
        ptr[i * 14 + 7] = g_buffers->shapeRotations[i].y;
        ptr[i * 14 + 8] = g_buffers->shapeRotations[i].z;
        ptr[i * 14 + 9] = g_buffers->shapeRotations[i].w;

        ptr[i * 14 + 10] = g_buffers->shapePrevRotations[i].x;
        ptr[i * 14 + 11] = g_buffers->shapePrevRotations[i].y;
        ptr[i * 14 + 12] = g_buffers->shapePrevRotations[i].z;
        ptr[i * 14 + 13] = g_buffers->shapePrevRotations[i].w;
    }

    pyflex_UnmapShapeBuffers(g_buffers);

    return states;
}

void pyflex_set_shape_color(py::array_t<float> color)
{
    auto buf = color.request();
    auto ptr = (float *)buf.ptr;
    for (int i = 0; i < 3; ++i)
        g_shape_color[i] = ptr[i];
}

void pyflex_set_shape_states(py::array_t<float> states)
{
    pyflex_MapShapeBuffers(g_buffers);

    auto buf = states.request();
    auto ptr = (float *)buf.ptr;

    for (size_t i = 0; i < (size_t)g_buffers->shapePositions.size(); i++)
    {
        g_buffers->shapePositions[i].x = ptr[i * 14];
        g_buffers->shapePositions[i].y = ptr[i * 14 + 1];
        g_buffers->shapePositions[i].z = ptr[i * 14 + 2];

        g_buffers->shapePrevPositions[i].x = ptr[i * 14 + 3];
        g_buffers->shapePrevPositions[i].y = ptr[i * 14 + 4];
        g_buffers->shapePrevPositions[i].z = ptr[i * 14 + 5];

        g_buffers->shapeRotations[i].x = ptr[i * 14 + 6];
        g_buffers->shapeRotations[i].y = ptr[i * 14 + 7];
        g_buffers->shapeRotations[i].z = ptr[i * 14 + 8];
        g_buffers->shapeRotations[i].w = ptr[i * 14 + 9];

        g_buffers->shapePrevRotations[i].x = ptr[i * 14 + 10];
        g_buffers->shapePrevRotations[i].y = ptr[i * 14 + 11];
        g_buffers->shapePrevRotations[i].z = ptr[i * 14 + 12];
        g_buffers->shapePrevRotations[i].w = ptr[i * 14 + 13];
    }

    UpdateShapes();

    pyflex_UnmapShapeBuffers(g_buffers);
}

py::array_t<float> pyflex_get_sceneUpper()
{
    auto scene_upper = py::array_t<float>(3);
    auto buf = scene_upper.request();
    auto ptr = (float *)buf.ptr;

    ptr[0] = g_sceneUpper.x;
    ptr[1] = g_sceneUpper.y;
    ptr[2] = g_sceneUpper.z;

    return scene_upper;
}

py::array_t<float> pyflex_get_sceneLower()
{
    auto scene_lower = py::array_t<float>(3);
    auto buf = scene_lower.request();
    auto ptr = (float *)buf.ptr;

    ptr[0] = g_sceneLower.x;
    ptr[1] = g_sceneLower.y;
    ptr[2] = g_sceneLower.z;

    return scene_lower;
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int>, py::float_> pyflex_get_camera_params()
{
    auto camPos = py::array_t<float>(3);
    auto camPosPtr = (float *)camPos.request().ptr;
    camPosPtr[0] = g_camPos.x;
    camPosPtr[1] = g_camPos.y;
    camPosPtr[2] = g_camPos.z;

    auto camAngle = py::array_t<float>(3);
    auto camAnglePtr = (float *)camAngle.request().ptr;
    camAnglePtr[0] = g_camAngle.x;
    camAnglePtr[1] = g_camAngle.y;
    camAnglePtr[2] = g_camAngle.z;

    auto camSize = py::array_t<int>(2);
    auto camSizePtr = (int *)camSize.request().ptr;
    camSizePtr[0] = g_screenWidth;
    camSizePtr[1] = g_screenHeight;

    return std::make_tuple(camPos, camAngle, camSize, py::float_(fov));
}

// std::map<py::str, py::object> pyflex_get_camera_params()
// {
//     std::map<py::str, py::object> dict;
//     dict[py::str("fov")] = py::float_(1.23);
//     return dict;
// }

void pyflex_set_camera_params(py::dict scene_params)
{
    if (g_render)
    {
        for (auto item : scene_params)
        {
            string key = py::str(item.first);
            if (key == "render_type")
            {
                g_drawPoints = false;
                g_drawCloth = false;
                for (auto it = item.second.begin(); it != py::iterator::sentinel(); it++)
                {
                    std::string render_type = py::str(*it);
                    if (render_type == "points")
                        g_drawPoints = true;
                    if (render_type == "cloth")
                        g_drawCloth = true;
                }
            }
            if (key == "cam_position")
            {
                auto it = item.second.begin();
                float cam_x = std::stof(py::str(*it));
                it++;
                float cam_y = std::stof(py::str(*it));
                it++;
                float cam_z = std::stof(py::str(*it));
                it++;
                g_camPos = Vec3(cam_x, cam_y, cam_z);
            }
            if (key == "cam_angle")
            {
                auto it = item.second.begin();
                float cam_angle_x = std::stof(py::str(*it));
                it++;
                float cam_angle_y = std::stof(py::str(*it));
                it++;
                float cam_angle_z = std::stof(py::str(*it));
                it++;
                g_camAngle = Vec3(cam_angle_x, cam_angle_y, cam_angle_z);
            }
            if (key == "cam_size")
            {
                auto it = item.second.begin();
                int cam_width = std::stoi(py::str(*it));
                it++;
                int cam_height = std::stoi(py::str(*it));
                it++;
                g_screenHeight = cam_height;
                g_screenWidth = cam_width;
            }
            if (key == "cam_fov")
            {
                fov = std::stof(py::str(item.second));
            }
        }
    }
}

std::tuple<
    py::array_t<unsigned char>,
    py::array_t<float>,
    py::array_t<float>>
pyflex_render(bool uv)
{
    MapBuffers(g_buffers);
    StartFrame(Vec4(g_clearColor, 1.0f));

    // main scene render
    RenderScene();
    RenderDebug();

    int newScene = DoUI();
    EndFrame();
    int rendered_img_int32_ptr[g_screenWidth * g_screenHeight];
    ReadFrame(rendered_img_int32_ptr, g_screenWidth, g_screenHeight);
    float rendered_depth_float_ptr[g_screenWidth * g_screenHeight];
    ReadDepth(rendered_depth_float_ptr, g_screenWidth, g_screenHeight);

    // now render UVs
    float *rendered_uvs_red_float_ptr = new float[g_screenWidth * g_screenHeight];
    float *rendered_uvs_green_float_ptr = new float[g_screenWidth * g_screenHeight];
    float *rendered_uvs_blue_float_ptr = new float[g_screenWidth * g_screenHeight];
    if (uv)
    {
        StartFrame(Vec4(g_clearColor, 0.0f));
        RenderScene(true);
        EndFrame();
        // float rendered_uvs_alpha_float_ptr[g_screenWidth * g_screenHeight];
        ReadFrame(rendered_uvs_red_float_ptr,
                rendered_uvs_green_float_ptr,
                rendered_uvs_blue_float_ptr,
                g_screenWidth,
                g_screenHeight);
    }

    auto rendered_img = py::array_t<uint8_t>((int)g_screenWidth * g_screenHeight * 4);
    auto rendered_img_ptr = (uint8_t *)rendered_img.request().ptr;

    auto rendered_depth = py::array_t<float>((float)g_screenWidth * g_screenHeight);
    auto rendered_depth_ptr = (float *)rendered_depth.request().ptr;

    auto rendered_uvs = py::array_t<float>((float)g_screenWidth * g_screenHeight * 3);
    auto rendered_uvs_ptr = (float *)rendered_uvs.request().ptr;

    for (int i = 0; i < g_screenWidth * g_screenHeight; ++i)
    {
        int32_abgr_to_int8_rgba((uint32_t)rendered_img_int32_ptr[i],
                                rendered_img_ptr[4 * i],
                                rendered_img_ptr[4 * i + 1],
                                rendered_img_ptr[4 * i + 2],
                                rendered_img_ptr[4 * i + 3]);
        rendered_depth_ptr[i] = 2 * g_camFar * g_camNear / (g_camFar + g_camNear - (2 * rendered_depth_float_ptr[i] - 1) * (g_camFar - g_camNear));
        if (uv)
        {
            rendered_uvs_ptr[3 * i] = rendered_uvs_red_float_ptr[i];
            rendered_uvs_ptr[3 * i + 1] = rendered_uvs_green_float_ptr[i];
            rendered_uvs_ptr[3 * i + 2] = rendered_uvs_blue_float_ptr[i];
            // rendered_uvs_ptr[4 * i + 3] = rendered_uvs_alpha_float_ptr[i];
        }
    }
    delete[] rendered_uvs_red_float_ptr;
    delete[] rendered_uvs_green_float_ptr;
    delete[] rendered_uvs_blue_float_ptr;

    UnmapBuffers(g_buffers);

    return std::make_tuple(
        rendered_img,
        rendered_depth,
        rendered_uvs);

}

PYBIND11_MODULE(pyflex, m)
{
    m.def("main", &main);
    m.def("init", &pyflex_init);
    m.def("set_scene", &pyflex_set_scene, "Set Scene");
    m.def("clean", &pyflex_clean);
    m.def("step", &pyflex_step,
          py::arg("update_params") = nullptr,
          py::arg("capture") = 0,
          py::arg("path") = nullptr);
    m.def("render", &pyflex_render, py::arg("uv")=false);

    m.def("get_camera_params", &pyflex_get_camera_params, "Get camera parameters");
    m.def("set_camera_params", &pyflex_set_camera_params, "Set camera parameters");

    m.def("add_box", &pyflex_add_box,
          py::arg("halfEdge_") = 0,
          py::arg("center_") = 0,
          py::arg("quat_") = 0,
          py::arg("trigger") = 0,
          "Add box to the scene");
    m.def("add_sphere", &pyflex_add_sphere, "Add sphere to the scene");
    m.def("add_capsule", &pyflex_add_capsule, "Add capsule to the scene");

    m.def("pop_shape", &pyflex_pop_shape, "remove shape from the scene");

    m.def("get_n_particles", &pyflex_get_n_particles, "Get the number of particles");
    m.def("get_n_shapes", &pyflex_get_n_shapes, "Get the number of shapes");
    m.def("get_n_rigids", &pyflex_get_n_rigids, "Get the number of rigids");
    m.def("get_n_rigidPositions", &pyflex_get_n_rigidPositions, "Get the number of rigid positions");

    m.def("get_phases", &pyflex_get_phases, "Get particle phases");
    m.def("set_phases", &pyflex_set_phases, "Set particle phases");
    m.def("get_groups", &pyflex_get_groups, "Get particle groups");
    m.def("set_groups", &pyflex_set_groups, "Set particle groups");

    m.def("get_positions", &pyflex_get_positions, "Get particle positions");
    m.def("set_positions", &pyflex_set_positions, "Set particle positions");

    m.def("get_edges", &pyflex_get_edges, "Get mesh edges");
    m.def("get_faces", &pyflex_get_faces, "Get mesh faces");

    m.def("get_restPositions", &pyflex_get_restPositions, "Get particle restPositions");
    m.def("get_rigidOffsets", &pyflex_get_rigidOffsets, "Get rigid offsets");
    m.def("get_rigidIndices", &pyflex_get_rigidIndices, "Get rigid indices");
    m.def("get_rigidLocalPositions", &pyflex_get_rigidLocalPositions, "Get rigid local positions");
    m.def("get_rigidGlobalPositions", &pyflex_get_rigidGlobalPositions, "Get rigid global positions");
    m.def("get_rigidRotations", &pyflex_get_rigidRotations, "Get rigid rotations");
    m.def("get_rigidTranslations", &pyflex_get_rigidTranslations, "Get rigid translations");

    // m.def("get_sceneParams", &pyflex_get_sceneParams, "Get scene parameters");

    m.def("get_velocities", &pyflex_get_velocities, "Get particle velocities");
    m.def("set_velocities", &pyflex_set_velocities, "Set particle velocities");

    m.def("get_shape_states", &pyflex_get_shape_states, "Get shape states");
    m.def("set_shape_states", &pyflex_set_shape_states, "Set shape states");
    m.def("clear_shapes", &ClearShapes, "Clear shapes");

    m.def("get_scene_upper", &pyflex_get_sceneUpper);
    m.def("get_scene_lower", &pyflex_get_sceneLower);

    m.def("add_rigid_body", &pyflex_add_rigid_body);
    m.def("set_shape_color", &pyflex_set_shape_color, "Set the color of the shape");

    m.def("add_cloth_square", &pyflex_add_cloth_square, "Add cloth (square)");
    m.def("add_cloth_mesh",
          &pyflex_add_cloth_mesh,
          "Add cloth (mesh)",
          py::arg("position"),
          py::arg("verts"),
          py::arg("faces"),
          py::arg("stretch_edges"),
          py::arg("bend_edges"),
          py::arg("shear_edges"),
          py::arg("uvs"),
          py::arg("stiffness"),
          py::arg("mass") = 1);
    m.def("emit_particles_box", &pyflex_emit_particles_box, "Emit particles (box)");
    m.def("emit_particles_cone", &pyflex_emit_particles_cone, "Emit particles (cone)");
    m.def("change_cloth_color", &pyflex_change_cloth_color, "Change color");
}
