# =============================================================================
# MFMA Course — CMakeLists.txt
# =============================================================================
#
# SYFTE (PURPOSE):
#   Konfigurerar byggmiljön för HIP-kärnor riktade mot AMD MI300X (gfx942).
#   Configures the build environment for HIP kernels targeting AMD MI300X (gfx942).
#
#   Denna fil använder CMakes INBYGGDA HIP-språkstöd (från version 3.21+).
#   This file uses CMake's NATIVE HIP language support (from version 3.21+).
#
#   Det finns TVÅ sätt att bygga HIP-kod med CMake:
#   There are TWO ways to build HIP code with CMake:
#
#     1. "HIP as a Language"  — project(... LANGUAGES HIP)
#        CMake förstår HIP som ett förstklassigt språk, precis som C eller CXX.
#        CMake understands HIP as a first-class language, just like C or CXX.
#        ✅ Vi använder detta (We use this approach).
#
#     2. "HIP as a Library"   — find_package(hip) + hip::device
#        Behandlar HIP som ett externt bibliotek. Äldre metod, mindre IDE-stöd.
#        Treats HIP as an external library. Older method, less IDE support.
#        ❌ Undvik detta för nya projekt (Avoid this for new projects).
#
#   VARFÖR SPELAR DET ROLL FÖR CLION? (WHY DOES THIS MATTER FOR CLION?)
#   CLion analyserar CMakeLists.txt för att förstå projektet. När HIP deklareras
#   som ett språk, vet CLion vilken kompilator som ska användas, vilka
#   inkluderingssökvägar som behövs, och kan generera korrekt compile_commands.json.
#   CLion parses CMakeLists.txt to understand the project. When HIP is declared
#   as a language, CLion knows which compiler to use, which include paths are
#   needed, and can generate the correct compile_commands.json.
#
# =============================================================================

# ---------------------------------------------------------------------------
# CMake 3.21 krävs för HIP-språkstöd.
# CMake 3.21 is required for HIP language support.
#
# cmake_policy(VERSION ...) sätter policyintervallet, dvs. vi accepterar
# beteenden från 3.21 till 3.30. Detta förhindrar varningar från nyare CMake
# versioner och säkerställer framåtkompatibilitet.
# cmake_policy(VERSION ...) sets the policy range, i.e. we accept behaviours
# from 3.21 to 3.30. This prevents warnings from newer CMake versions and
# ensures forward compatibility.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.21)
cmake_policy(VERSION 3.21...3.30)


# ---------------------------------------------------------------------------
# PROJEKT-DEKLARATION (PROJECT DECLARATION)
#
# LANGUAGES CXX HIP — talar om för CMake att vi använder BÅDE C++ och HIP.
# LANGUAGES CXX HIP — tells CMake we use BOTH C++ and HIP.
#
# Varför båda? (Why both?)
#   - CXX:  för eventuella rena värdkodfiler (for any pure host-code files)
#   - HIP:  för kärnor och enhetskod (for kernels and device code)
#
# När CMake ser "HIP" i LANGUAGES, gör den följande:
# When CMake sees "HIP" in LANGUAGES, it does the following:
#
#   1. Söker efter hip-lang-config.cmake i ROCm-installationen
#      Searches for hip-lang-config.cmake in the ROCm installation
#
#   2. Ställer in CMAKE_HIP_COMPILER (vanligtvis /opt/rocm/bin/hipcc)
#      Sets CMAKE_HIP_COMPILER (typically /opt/rocm/bin/hipcc)
#
#   3. Detekterar GPU-arkitekturer via rocm_agent_enumerator
#      Detects GPU architectures via rocm_agent_enumerator
#
#   4. Definierar regler för kompilering av .hip-filer automatiskt
#      Defines rules for compiling .hip files automatically
#
# VERSIONSHANTERING (VERSIONING):
#   Vi sätter VERSION här så att CLion visar det i projektöversikten.
#   We set VERSION here so CLion displays it in the project overview.
# ---------------------------------------------------------------------------
project(mfma_course
        VERSION 0.6.0
        DESCRIPTION "MFMA Kernel Engineering Course — CDNA3 / MI300X"
        LANGUAGES CXX HIP
)


# ---------------------------------------------------------------------------
# GPU-ARKITEKTUR (GPU ARCHITECTURE)
#
# CMAKE_HIP_ARCHITECTURES styr vilken GPU-kod som genereras.
# CMAKE_HIP_ARCHITECTURES controls which GPU code is generated.
#
# gfx942 = AMD Instinct MI300X (CDNA3)
#   - 8 XCD:er (Accelerator Complex Dies), 304 beräkningsenheter
#   - 8 XCDs (Accelerator Complex Dies), 304 compute units
#   - MFMA v3-instruktioner (MFMA v3 instructions)
#   - 192 GB HBM3, 5.3 TB/s bandbredd (bandwidth)
#
# VIKTIGT FÖR CLION (IMPORTANT FOR CLION):
#   Om du inte sätter detta explicit, försöker CMake köra
#   rocm_agent_enumerator på VÄRDDATORN under konfigurering.
#   If you don't set this explicitly, CMake tries to run
#   rocm_agent_enumerator on the HOST MACHINE during configuration.
#
#   Vid fjärrutveckling (remote development) kan detta misslyckas om
#   CMake konfigureras lokalt innan synkronisering med fjärrvärden.
#   During remote development, this can fail if CMake configures
#   locally before syncing with the remote host.
#
#   Sätter vi det explicit undviker vi detekteringsproblem.
#   Setting it explicitly avoids detection issues.
#
# ANDRA ARKITEKTURER (OTHER ARCHITECTURES):
#   gfx90a  = MI250X / MI210 (CDNA2)
#   gfx908  = MI100 (CDNA1)
#   gfx1100 = RX 7900 XTX (RDNA3) — OBS: saknar MFMA, använder WMMA
#   gfx803  = RX 580 (GCN/Polaris) — OBS: inga matrisinstruktioner alls
#                                     NOTE: no matrix instructions at all
# ---------------------------------------------------------------------------
set(CMAKE_HIP_ARCHITECTURES gfx942 CACHE STRING
        "Target GPU architecture(s) — gfx942 for MI300X CDNA3")


# ---------------------------------------------------------------------------
# KOMPILATORNS FLAGGOR (COMPILER FLAGS)
#
# Vi definierar flaggor per byggtyp (build type). Detta ger CLion möjlighet
# att växla mellan Debug/Release/RelWithDebInfo via CMake-profiler.
# We define flags per build type. This gives CLion the ability to switch
# between Debug/Release/RelWithDebInfo via CMake profiles.
#
# FLAGG-FÖRKLARING (FLAG EXPLANATION):
#
#   -O3              Maximal optimering. Kompilatorn kan ändra ordningen på
#                    instruktioner, veckla ut loopar, vektorisera.
#                    Maximum optimisation. The compiler may reorder
#                    instructions, unroll loops, vectorise.
#
#   -O0              Ingen optimering. Varje rad C++ → en tydlig sekvens
#                    av assemblerinstruktioner. Nödvändigt för felsökning.
#                    No optimisation. Each C++ line → a clear sequence of
#                    assembly instructions. Necessary for debugging.
#
#   -g               Genererar felsökningsinformation (DWARF-format).
#                    Generates debug information (DWARF format).
#                    ROCgdb kan sedan stega genom kärnkod.
#                    ROCgdb can then step through kernel code.
#
#   -save-temps      Sparar mellanliggande filer: .ll (LLVM IR), .s (assembler).
#                    Saves intermediate files: .ll (LLVM IR), .s (assembly).
#                    Ovärderligt för att studera vad kompilatorn gör med din kod.
#                    Invaluable for studying what the compiler does with your code.
#
#   -Rpass=.*        Rapporterar kompilatorns optimeringar till stderr.
#                    Reports the compiler's optimisation passes to stderr.
#                    Användbart för att förstå vad som inlinade, vad som inte
#                    gjorde det, och varför.
#                    Useful for understanding what inlined, what didn't, and why.
#
#   -munsafe-fp-atomics  Tillåter snabbare atomära flyttalsoperationer.
#                        Allows faster atomic floating-point operations.
#                        MI300X har hårdvarustöd, men resultatet kan variera
#                        i sista biten (ULP). Acceptabelt för ML-arbetsbelastningar.
#                        MI300X has hardware support, but results may vary in
#                        the last bit (ULP). Acceptable for ML workloads.
# ---------------------------------------------------------------------------

# --- Release: maximal prestanda (maximum performance) ---
set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG -munsafe-fp-atomics" CACHE STRING
        "HIP flags for Release builds")

# --- Debug: full insyn, ingen optimering (full visibility, no optimisation) ---
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -g -save-temps" CACHE STRING
        "HIP flags for Debug builds")

# --- RelWithDebInfo: optimerat men felsökningsbart (optimised but debuggable) ---
# Detta är den BÄSTA profilen för daglig utveckling i CLion.
# This is the BEST profile for daily development in CLion.
# Du får bra prestanda OCH kan stega genom koden med ROCgdb.
# You get good performance AND can step through code with ROCgdb.
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG" CACHE STRING
        "HIP flags for RelWithDebInfo builds")

# --- Gemensamma flaggor för ALLA byggtyper (common flags for ALL build types) ---
# Dessa läggs till oavsett Debug/Release-val.
# These are appended regardless of Debug/Release choice.
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -D__HIP_PLATFORM_AMD__=" CACHE STRING
        "Common HIP compiler flags" FORCE)


# ---------------------------------------------------------------------------
# KOMPILERA DATABASGENERERING (COMPILATION DATABASE GENERATION)
#
# compile_commands.json — detta är NYCKELN till CLions intelligens.
# compile_commands.json — this is the KEY to CLion's intelligence.
#
# Utan denna fil: CLion gissar inkluderingssökvägar, visar falska fel,
#   och kan inte navigera till HIP-definitioner.
# Without this file: CLion guesses include paths, shows false errors,
#   and cannot navigate to HIP definitions.
#
# Med denna fil: CLion vet EXAKT vilka flaggor som används för varje fil,
#   inklusive --offload-arch=gfx942, alla -I sökvägar, alla -D makron.
# With this file: CLion knows EXACTLY which flags are used for each file,
#   including --offload-arch=gfx942, all -I paths, all -D macros.
#
# CLion läser denna fil automatiskt från byggkatalogen.
# CLion reads this file automatically from the build directory.
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ---------------------------------------------------------------------------
# KÄLL-FILERS SPRÅKÖVERSTYRNING (SOURCE FILE LANGUAGE OVERRIDE)
#
# PROBLEMET (THE PROBLEM):
#   Våra källfiler har .cpp-ändelse, men innehåller HIP-kärnor (__global__,
#   hipMalloc, MFMA-intrinsics, etc.). CMakes HIP-språk associerar
#   automatiskt BARA .hip-filer med HIP-kompilatorn.
#   Our source files have .cpp extension, but contain HIP kernels (__global__,
#   hipMalloc, MFMA intrinsics, etc.). CMake's HIP language automatically
#   associates ONLY .hip files with the HIP compiler.
#
# LÖSNINGEN (THE SOLUTION):
#   set_source_files_properties(... PROPERTIES LANGUAGE HIP) tvingar CMake
#   att kompilera dessa .cpp-filer med hipcc istället för g++/clang++.
#   set_source_files_properties(... PROPERTIES LANGUAGE HIP) forces CMake
#   to compile these .cpp files with hipcc instead of g++/clang++.
#
# ALTERNATIV (ALTERNATIVE):
#   Byt namn på alla filer till .hip — men detta bryter befintligt arbetsflöde
#   med Makefile och manuella hipcc-kommandon. Inte värt besväret ännu.
#   Rename all files to .hip — but this breaks the existing workflow with
#   Makefile and manual hipcc commands. Not worth the trouble yet.
#
# Vi samlar alla HIP-källfiler i en lista för att undvika upprepning.
# We collect all HIP source files in a list to avoid repetition.
# ---------------------------------------------------------------------------
set(HIP_SOURCES
        01-hello-hip/hello_hip.cpp
        02-wavefront-basics/wavefront_basics.cpp
        03-lds-memory/lds_memory.cpp
        04-mfma-intro/mfma_intro.cpp
        05-mfma-gemm/mfma_gemm.cpp
        06-xcd-awareness/xcd_discovery.cpp
)

set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)


# ---------------------------------------------------------------------------
# EXEKVERBARA MÅL (EXECUTABLE TARGETS)
#
# Varje experiment är ett separat mål (target) i CMake. Detta innebär att
# CLion visar dem i konfigurationsväljaren (dropdown uppe till höger).
# Each experiment is a separate target in CMake. This means CLion shows
# them in the configuration selector (dropdown at the top right).
#
# Du kan:
# You can:
#   - Välja "hello_hip" och trycka Ctrl+Shift+F10 för att bygga+köra
#     Select "hello_hip" and press Ctrl+Shift+F10 to build+run
#   - Välja "mfma_gemm" och sätta brytpunkter (breakpoints) med ROCgdb
#     Select "mfma_gemm" and set breakpoints with ROCgdb
#   - Köra "all_experiments" för att bygga allt på en gång
#     Run "all_experiments" to build everything at once
# ---------------------------------------------------------------------------
add_executable(hello_hip           01-hello-hip/hello_hip.cpp)
add_executable(wavefront_basics    02-wavefront-basics/wavefront_basics.cpp)
add_executable(lds_memory          03-lds-memory/lds_memory.cpp)
add_executable(mfma_intro          04-mfma-intro/mfma_intro.cpp)
add_executable(mfma_gemm           05-mfma-gemm/mfma_gemm.cpp)
add_executable(xcd_discovery       06-xcd-awareness/xcd_discovery.cpp)


# ---------------------------------------------------------------------------
# MÅL-SPECIFIKA INKLUDERINGSSÖKVÄGAR (TARGET-SPECIFIC INCLUDE PATHS)
#
# Varje mål kan behöva extra inkluderingssökvägar. Här sätter vi den
# gemensamma ROCm-sökvägen explicit, som en reserv (fallback) ifall
# CMake inte hittar den automatiskt.
# Each target may need extra include paths. Here we set the common ROCm
# path explicitly, as a fallback in case CMake doesn't find it automatically.
#
# PRIVATE = denna sökväg gäller BARA vid kompilering av detta mål,
#   inte för mål som beror på (link to) detta mål.
# PRIVATE = this path applies ONLY when compiling this target,
#   not for targets that depend on (link to) this target.
# ---------------------------------------------------------------------------
set(ALL_TARGETS hello_hip wavefront_basics lds_memory mfma_intro mfma_gemm)

foreach(target IN LISTS ALL_TARGETS)
    target_include_directories(${target} PRIVATE
            ${CMAKE_SOURCE_DIR}
    )
endforeach()


# ---------------------------------------------------------------------------
# BEKVÄMLIGHETS-MÅL (CONVENIENCE TARGET)
#
# "all_experiments" bygger alla experiment med ett enda kommando.
# Användbart i CLion: välj detta mål och tryck Build (Ctrl+F9).
# "all_experiments" builds all experiments with a single command.
# Useful in CLion: select this target and press Build (Ctrl+F9).
# ---------------------------------------------------------------------------
add_custom_target(all_experiments
        DEPENDS ${ALL_TARGETS}
        COMMENT "Bygger alla experiment / Building all experiments"
)


# ---------------------------------------------------------------------------
# ASSEMBLER-GENERERING (ASSEMBLY GENERATION)
#
# Dessa anpassade mål skapar assemblerfiler (.s) för varje experiment.
# These custom targets create assembly files (.s) for each experiment.
#
# VARFÖR? (WHY?)
#   När du skriver MFMA-kärnor måste du verifiera att kompilatorn genererar
#   rätt instruktioner: v_mfma_f32_32x32x8f16, ds_read_b128, etc.
#   When writing MFMA kernels you must verify the compiler generates the
#   correct instructions: v_mfma_f32_32x32x8f16, ds_read_b128, etc.
#
#   I CLion: välj t.ex. "asm_mfma_gemm" som mål, bygg det, och öppna
#   sedan .s-filen direkt i editorn.
#   In CLion: select e.g. "asm_mfma_gemm" as the target, build it, then
#   open the .s file directly in the editor.
# ---------------------------------------------------------------------------
foreach(target IN LISTS ALL_TARGETS)
    # Hämta källfilen som tillhör målet
    # Get the source file belonging to the target
    get_target_property(target_sources ${target} SOURCES)
    list(GET target_sources 0 source_file)

    # Bestäm utdatafilens namn (determine the output filename)
    get_filename_component(source_dir ${source_file} DIRECTORY)
    get_filename_component(source_name ${source_file} NAME_WE)

    add_custom_target(asm_${target}
            COMMAND ${CMAKE_HIP_COMPILER}
            --offload-arch=${CMAKE_HIP_ARCHITECTURES}
            -O3 -S
            -o ${CMAKE_SOURCE_DIR}/${source_dir}/${source_name}.s
            ${CMAKE_SOURCE_DIR}/${source_file}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Genererar assembler för ${target} / Generating assembly for ${target}"
            VERBATIM
    )
endforeach()


# ---------------------------------------------------------------------------
# STATUSRAPPORT (STATUS REPORT)
#
# Skriver ut konfigurationsinformation vid CMake-körning.
# Prints configuration information during CMake run.
# CLion visar detta i "CMake"-fliken längst ned.
# CLion shows this in the "CMake" tab at the bottom.
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║         MFMA Course — Konfigurationssammanfattning       ║")
message(STATUS "║         MFMA Course — Configuration Summary              ║")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  Byggtyp / Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "║  GPU-arkitektur / GPU arch: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "║  HIP-kompilator / compiler: ${CMAKE_HIP_COMPILER}")
message(STATUS "║  HIP-flaggor / HIP flags:   ${CMAKE_HIP_FLAGS}")
message(STATUS "║  Compile DB:               ${CMAKE_EXPORT_COMPILE_COMMANDS}")
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "")