<img width="1558" height="876" alt="image" src="https://github.com/user-attachments/assets/3f12e047-533f-48b2-9812-27c7fba4f8f3" />

ArmorPaint - Bake edition
==============

> **Unofficial fork** of [ArmorPaint](https://github.com/armory3d/armorpaint) by DAVECHAG. It is not reviewed or supported by the ArmorPaint developer. Official builds, manual and support: [armorpaint.org](https://armorpaint.org).

[ArmorPaint](https://armorpaint.org) is a software for 3D PBR texture painting - check out the [manual](https://armorpaint.org/manual).

**This fork** adds a mesh map baker (13 maps, GPU or CPU, high poly projection, Cavity, normal map, ambient occlusion, height etc...), an Inpaint tool to repair bake artifacts and a new brush bar: see [docs/bake](docs/bake/README.md).

*Note 1 (from the official project): the [ArmorPaint repository](https://github.com/armory3d/armorpaint) is aimed at developers and may not be stable. Distributed binaries are [paid](https://armorpaint.org/download) to help with the project funding. Please support the original project!*

*Note 2: If you are compiling git version of ArmorPaint, then you need to have a compiler ([Visual Studio with clang tools](https://visualstudio.microsoft.com/downloads/) - Windows, [clang + dependencies](https://github.com/armory3d/armorpaint/blob/main/base/docs/linux_deps.md) - Linux, [Xcode](https://developer.apple.com/xcode/resources/) - macOS / iOS, [Android Studio](https://developer.android.com/studio) - Android) and [git](https://git-scm.com/downloads) installed.*

```bash
git clone -b feature/bake-dialog https://github.com/DAVECHAG/Armarpaint-bake
cd Armarpaint-bake/paint
```

*For the official version without the bake system: `git clone https://github.com/armory3d/armorpaint`.*

**Windows (x64)**
```bash
..\base\make
# Open generated Visual Studio project at `build\ArmorPaint.sln`
# Build and run
```

**Linux (x64)**
```bash
../base/make --run
```

**macOS (arm64)**
```bash
../base/make
# Open generated Xcode project at `build/ArmorPaint.xcodeproj`
# Build and run
```

**Android (arm64)**
```bash
../base/make --target android
# Open generated Android Studio project at `build/ArmorPaint`
# Build for device
```

**iOS (arm64)**
```bash
../base/make --target ios
# Open generated Xcode project `build/ArmorPaint.xcodeproj`
# Build for device
```

**WASM**
```bash
../base/make --target wasm --compile --embed
```

**Embedding data files**
```bash
# Requires compiler with c23 #embed support (clang 19 or newer)
../base/make --embed
```
