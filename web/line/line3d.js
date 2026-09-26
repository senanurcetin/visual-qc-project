// 3D digital twin of the inspection line on the HMI dashboard.
// Renders a conveyor, a QC scan gantry, a reject pusher and an OK stacking pallet with
// Three.js, and keeps its own clock in sync with the simulation through /api/data.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

// ── Line geometry (metres) and the cycle timeline (fractions of one 4 s cycle) ──
const BELT = { start: -7.5, end: 7.5, top: 0.9, width: 1.8 };
const PLATE = { l: 1.6, h: 0.08, w: 1.1 };
const PLATE_Y = BELT.top + PLATE.h / 2;
const GANTRY_X = -1;
const DIVERTER_X = 3.5;
const PALLET = { x: 9.1, top: 0.2 };
const BIN = { x: DIVERTER_X, z: 2.6, floor: 0.28 };
const xAt = (p) => -7 + 16 * p; // belt travel: 4 m/s at a 4 s cycle
const PHASE = {
    scanStart: 0.325, scanEnd: 0.425, verdict: 0.43,
    pushStart: (DIVERTER_X + 7) / 16, pushEnd: 0.74, fallEnd: 0.8, retractEnd: 0.84,
    beltEnd: (BELT.end + 7) / 16, palletLand: 0.955, land: 0.97,
};
const PALLET_MAX = 12;
const PUSHER_Z = -1.7;      // pusher body (world z)
const PADDLE_REST_Z = -1.15; // retracted paddle, just outside the rail gap
const ROD_BASE_Z = -0.14;    // front face of the pusher body, in group space
const BIN_MAX = 8;
const POLL_MS = 300;
const COLORS = { ok: 0x22c55e, fail: 0xef4444, scan: 0x38bdf8, amber: 0xf59e0b };

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => t * t * (3 - 2 * t);

// ── Procedural textures ──────────────────────────────────────────────────────
function rng(seed) {
    return () => {
        seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function brushedSteel(ctx, w, h, rand) {
    ctx.fillStyle = "#a7b0ba";
    ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < 900; i++) {
        const y = rand() * h;
        const shade = 150 + Math.floor(rand() * 70);
        ctx.strokeStyle = `rgba(${shade},${shade + 6},${shade + 12},${0.08 + rand() * 0.12})`;
        ctx.lineWidth = 0.5 + rand();
        ctx.beginPath();
        const x0 = rand() * w * 0.6;
        ctx.moveTo(x0, y);
        ctx.lineTo(x0 + w * (0.3 + rand() * 0.7), y + (rand() - 0.5) * 1.5);
        ctx.stroke();
    }
    const edge = ctx.createLinearGradient(0, 0, 0, h);
    edge.addColorStop(0, "rgba(255,255,255,0.08)");
    edge.addColorStop(0.5, "rgba(0,0,0,0)");
    edge.addColorStop(1, "rgba(0,0,0,0.12)");
    ctx.fillStyle = edge;
    ctx.fillRect(0, 0, w, h);
}

const DEFECT_PAINTERS = {
    scratches(ctx, w, h, rand) {
        for (let i = 0; i < 5; i++) {
            const y = h * (0.15 + rand() * 0.7);
            const x0 = w * rand() * 0.3, x1 = w * (0.6 + rand() * 0.4);
            const dy = (rand() - 0.5) * h * 0.25;
            ctx.strokeStyle = "rgba(30,34,40,0.85)"; ctx.lineWidth = 2.2 + rand() * 1.8;
            ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y + dy); ctx.stroke();
            ctx.strokeStyle = "rgba(245,248,252,0.7)"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(x0, y - 2); ctx.lineTo(x1, y + dy - 2); ctx.stroke();
        }
    },
    crazing(ctx, w, h, rand) {
        const cx = w * (0.3 + rand() * 0.4), cy = h * (0.3 + rand() * 0.4);
        ctx.strokeStyle = "rgba(25,28,34,0.8)";
        for (let i = 0; i < 70; i++) {
            let x = cx + (rand() - 0.5) * w * 0.5, y = cy + (rand() - 0.5) * h * 0.55;
            ctx.lineWidth = 0.8 + rand() * 1.2;
            ctx.beginPath(); ctx.moveTo(x, y);
            for (let k = 0; k < 5; k++) {
                x += (rand() - 0.5) * 34; y += (rand() - 0.5) * 34;
                ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
    },
    inclusion(ctx, w, h, rand) {
        for (let i = 0; i < 6; i++) {
            const x = w * (0.15 + rand() * 0.7), y = h * (0.15 + rand() * 0.7);
            const g = ctx.createRadialGradient(x, y, 1, x, y, 26);
            g.addColorStop(0, "rgba(20,22,26,0.95)"); g.addColorStop(1, "rgba(20,22,26,0)");
            ctx.save(); ctx.translate(x, y); ctx.scale(3.2, 0.55); ctx.translate(-x, -y);
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 26, 0, Math.PI * 2); ctx.fill();
            ctx.restore();
        }
    },
    patches(ctx, w, h, rand) {
        for (let i = 0; i < 5; i++) {
            const x = w * (0.1 + rand() * 0.8), y = h * (0.1 + rand() * 0.8), r = 30 + rand() * 60;
            const g = ctx.createRadialGradient(x, y, r * 0.2, x, y, r);
            g.addColorStop(0, "rgba(58,52,48,0.8)"); g.addColorStop(1, "rgba(58,52,48,0)");
            ctx.fillStyle = g;
            ctx.beginPath();
            for (let a = 0; a <= Math.PI * 2 + 0.01; a += Math.PI / 9) {
                const rr = r * (0.7 + rand() * 0.45);
                ctx.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
            }
            ctx.fill();
        }
    },
    pitted_surface(ctx, w, h, rand) {
        const cx = w * (0.3 + rand() * 0.4), cy = h * (0.3 + rand() * 0.4);
        for (let i = 0; i < 140; i++) {
            const x = cx + (rand() - 0.5) * w * 0.55, y = cy + (rand() - 0.5) * h * 0.6, r = 1.5 + rand() * 4.5;
            ctx.fillStyle = "rgba(240,244,248,0.5)";
            ctx.beginPath(); ctx.arc(x - 1, y - 1, r + 1, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = "rgba(18,20,24,0.9)";
            ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
        }
    },
    "rolled-in_scale"(ctx, w, h, rand) {
        for (let band = 0; band < 3; band++) {
            const y0 = h * (0.2 + rand() * 0.6);
            for (let i = 0; i < 45; i++) {
                const x = rand() * w, y = y0 + (rand() - 0.5) * 36;
                ctx.fillStyle = `rgba(${30 + rand() * 30},${26 + rand() * 20},${24 + rand() * 16},0.85)`;
                ctx.beginPath();
                ctx.moveTo(x, y);
                ctx.lineTo(x + 8 + rand() * 22, y + (rand() - 0.5) * 6);
                ctx.lineTo(x + 4 + rand() * 12, y + 3 + rand() * 5);
                ctx.closePath(); ctx.fill();
            }
        }
    },
};

function plateTexture(defect, seed) {
    const w = 512, h = 352;
    const canvas = document.createElement("canvas");
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext("2d");
    const rand = rng(seed);
    brushedSteel(ctx, w, h, rand);
    if (defect && DEFECT_PAINTERS[defect]) DEFECT_PAINTERS[defect](ctx, w, h, rand);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    return tex;
}

function beltTexture() {
    const canvas = document.createElement("canvas");
    canvas.width = 128; canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#171c24"; ctx.fillRect(0, 0, 128, 128);
    ctx.strokeStyle = "#252c37"; ctx.lineWidth = 10;
    ctx.beginPath(); ctx.moveTo(20, 0); ctx.lineTo(64, 64); ctx.lineTo(20, 128); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.03)";
    for (let i = 0; i < 300; i++) ctx.fillRect(Math.random() * 128, Math.random() * 128, 1, 1);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(BELT.end - BELT.start, 1);
    return tex;
}

// ── Scene construction ───────────────────────────────────────────────────────
function box(w, h, d, material, x = 0, y = 0, z = 0, parent) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent?.add(mesh);
    return mesh;
}

function glowMaterial(color, strength = 2.5, opacity = 1) {
    const m = new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, toneMapped: false });
    m.color.multiplyScalar(strength);
    return m;
}

function buildScene(scene) {
    const paint = new THREE.MeshStandardMaterial({ color: 0x3a4a5f, metalness: 0.55, roughness: 0.45 });
    const steel = new THREE.MeshStandardMaterial({ color: 0x8994a3, metalness: 0.9, roughness: 0.3 });
    const dark = new THREE.MeshStandardMaterial({ color: 0x1b222c, metalness: 0.4, roughness: 0.6 });
    const safety = new THREE.MeshStandardMaterial({ color: 0xeab308, metalness: 0.2, roughness: 0.6 });

    // Floor, grid and walkway markings
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 80),
        new THREE.MeshStandardMaterial({ color: 0x0b1220, metalness: 0.15, roughness: 0.82 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    const grid = new THREE.GridHelper(80, 80, 0x1e3a5f, 0x14263d);
    grid.material.transparent = true; grid.material.opacity = 0.35; grid.position.y = 0.002;
    scene.add(grid);
    for (const z of [-2.7, 3.7]) {
        const line = new THREE.Mesh(new THREE.PlaneGeometry(22, 0.08), safety);
        line.rotation.x = -Math.PI / 2; line.position.set(1, 0.004, z);
        scene.add(line);
    }

    // Conveyor: belt, rails, legs and end rollers
    const beltTex = beltTexture();
    const beltLen = BELT.end - BELT.start;
    const beltTopMat = new THREE.MeshStandardMaterial({ map: beltTex, metalness: 0.1, roughness: 0.85 });
    box(beltLen, 0.06, BELT.width, [dark, dark, beltTopMat, dark, dark, dark], 0, BELT.top - 0.03, 0, scene);
    // Side rails leave a gap at the diverter so the pusher and the rejected plate can cross.
    const railSegments = [[BELT.start - 0.25, DIVERTER_X - 0.9], [DIVERTER_X + 0.9, BELT.end + 0.25]];
    for (const z of [-1, 1]) {
        for (const [a, b] of railSegments) {
            box(b - a, 0.2, 0.08, paint, (a + b) / 2, BELT.top - 0.02, z * (BELT.width / 2 + 0.06), scene);
            box(b - a, 0.03, 0.09, glowMaterial(0x1d4ed8, 1.2), (a + b) / 2, BELT.top + 0.08, z * (BELT.width / 2 + 0.06), scene);
        }
        for (let x = BELT.start + 0.4; x <= BELT.end; x += 2.5) box(0.1, BELT.top - 0.1, 0.1, paint, x, (BELT.top - 0.1) / 2, z * 0.85, scene);
    }
    const rollers = [];
    for (const x of [BELT.start, BELT.end]) {
        const roller = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, BELT.width + 0.1, 24), steel);
        roller.rotation.x = Math.PI / 2; roller.position.set(x, BELT.top - 0.16, 0); roller.castShadow = true;
        scene.add(roller); rollers.push(roller);
    }

    // Feed hood: plates emerge from the pickling line
    const hood = new THREE.Group();
    box(3.1, 1.3, 2.3, paint, 0, 0.65, 0, hood);
    for (const z of [-1.12, 1.12]) box(3.12, 0.07, 0.08, safety, 0, 1.28, z, hood);
    box(0.08, 0.07, 2.32, safety, 1.52, 1.28, 0, hood);
    box(0.04, 0.5, 2.0, new THREE.MeshStandardMaterial({ color: 0x05080d, roughness: 1 }), 1.56, 0.22, 0, hood);
    hood.position.set(-6.35, BELT.top - 0.05, 0);
    scene.add(hood);

    // QC gantry with camera, ring light and laser sheet
    const gantry = new THREE.Group();
    for (const z of [-1.45, 1.45]) box(0.22, 3.2, 0.22, paint, 0, 1.6, z, gantry);
    box(0.36, 0.3, 3.2, paint, 0, 3.15, 0, gantry);
    box(0.52, 0.38, 0.5, dark, 0, 2.8, 0, gantry);
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.18, 24), steel);
    lens.position.set(0, 2.53, 0); gantry.add(lens);
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.025, 12, 48), glowMaterial(COLORS.scan, 2.2));
    ring.rotation.x = Math.PI / 2; ring.position.set(0, 2.45, 0); gantry.add(ring);
    const sheetMat = new THREE.MeshBasicMaterial({ color: COLORS.scan, transparent: true, opacity: 0.08, side: THREE.DoubleSide,
        depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false });
    const sheet = new THREE.Mesh(new THREE.PlaneGeometry(BELT.width, 2.45 - BELT.top), sheetMat);
    sheet.rotation.y = Math.PI / 2; sheet.position.set(0, (2.45 + BELT.top) / 2, 0); gantry.add(sheet);
    const beltLaser = box(0.03, 0.006, BELT.width, glowMaterial(COLORS.scan, 3), 0, BELT.top + 0.004, 0, gantry);
    beltLaser.castShadow = false;
    const spot = new THREE.SpotLight(0xcfeaff, 18, 5, Math.PI / 7, 0.5, 1.2);
    spot.position.set(0, 2.5, 0); spot.target.position.set(0, BELT.top, 0);
    gantry.add(spot, spot.target);
    gantry.position.x = GANTRY_X;
    scene.add(gantry);

    // Stack light tower (green = running, amber = paused, red = e-stop)
    const tower = new THREE.Group();
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 2.2, 12), steel);
    pole.position.y = 1.1; tower.add(pole);
    const lamps = {};
    [["green", 0x22c55e, 2.25], ["amber", 0xf59e0b, 2.45], ["red", 0xef4444, 2.65]].forEach(([name, color, y]) => {
        const mat = new THREE.MeshBasicMaterial({ color, toneMapped: false });
        const lamp = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.18, 20), mat);
        lamp.position.y = y; tower.add(lamp);
        lamps[name] = { mat, base: new THREE.Color(color) };
    });
    tower.position.set(GANTRY_X - 0.5, 0, -2.1);
    scene.add(tower);

    // Reject pusher and bin
    const pusher = new THREE.Group();
    box(1.3, 0.3, 0.32, paint, 0, 0, -0.3, pusher);
    const rod = box(0.06, 0.06, 1, steel, 0, 0, 0, pusher);
    const paddle = box(1.25, 0.24, 0.06, safety, 0, -0.08, 0, pusher);
    pusher.position.set(DIVERTER_X, BELT.top + 0.2, PUSHER_Z);
    scene.add(pusher);
    const binMat = new THREE.MeshStandardMaterial({ color: 0x7f1d1d, metalness: 0.45, roughness: 0.5 });
    box(2, 0.06, 1.6, binMat, BIN.x, BIN.floor - 0.03, BIN.z, scene);
    box(2, 0.55, 0.06, binMat, BIN.x, BIN.floor + 0.27, BIN.z + 0.8, scene);
    box(2, 0.3, 0.06, binMat, BIN.x, BIN.floor + 0.15, BIN.z - 0.8, scene);
    for (const dx of [-1, 1]) box(0.06, 0.55, 1.6, binMat, BIN.x + dx, BIN.floor + 0.27, BIN.z, scene);

    // OK pallet
    const wood = new THREE.MeshStandardMaterial({ color: 0x8a5a2c, roughness: 0.9, metalness: 0 });
    box(1.9, 0.12, 1.4, wood, PALLET.x, 0.14, 0, scene);
    for (const dz of [-0.55, 0, 0.55]) box(1.9, 0.08, 0.2, wood, PALLET.x, 0.04, dz, scene);

    return { beltTex, rollers, sheetMat, beltLaser, ring, lamps, rod, paddle };
}

// ── Main ─────────────────────────────────────────────────────────────────────
function init() {
    const container = document.getElementById("line-twin");
    if (!container) return;
    let renderer;
    try {
        renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    } catch (err) {
        console.warn("WebGL unavailable, using 2D feed", err);
        window.showLineFallback?.();
        return;
    }
    const rect = container.getBoundingClientRect();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
    renderer.setSize(rect.width, rect.height);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.domElement.className = "twin-canvas";
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute("aria-label", "3D view of the simulated inspection line: plates travel through a camera gantry, rejects are pushed into a bin, good plates are stacked on a pallet");
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070d18);
    scene.fog = new THREE.Fog(0x070d18, 22, 48);
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.55;
    scene.add(new THREE.HemisphereLight(0x9fb7d9, 0x0b1220, 0.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(6, 12, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -13, right: 13, top: 8, bottom: -8, near: 1, far: 40 });
    key.shadow.bias = -0.0005;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x7fb2ff, 0.7);
    rim.position.set(-8, 6, -10);
    scene.add(rim);

    const camera = new THREE.PerspectiveCamera(28, rect.width / rect.height, 0.1, 200);
    const controls = new OrbitControls(camera, renderer.domElement);
    Object.assign(controls, { enableDamping: true, enablePan: false, minDistance: 7, maxDistance: 44,
        minPolarAngle: 0.45, maxPolarAngle: 1.36, minAzimuthAngle: -1.35, maxAzimuthAngle: 0.7 });

    // Frame the whole line (feed hood → pallet) from a fixed 3/4 angle at any panel size.
    const VIEW_TARGET = new THREE.Vector3(1.2, 0.9, 0.4);
    const VIEW_DIR = new THREE.Vector3(-0.58, 0.5, 0.64).normalize();
    const lineBox = new THREE.Box3(new THREE.Vector3(-7.9, 0, -2.3), new THREE.Vector3(10.1, 3.4, 3.4));
    const corners = [];
    for (const x of [lineBox.min.x, lineBox.max.x]) for (const y of [lineBox.min.y, lineBox.max.y])
        for (const z of [lineBox.min.z, lineBox.max.z]) corners.push(new THREE.Vector3(x, y, z));
    const probe = new THREE.Vector3();
    function fitCamera() {
        controls.target.copy(VIEW_TARGET);
        for (let d = 7; d <= 44; d += 0.25) {
            camera.position.copy(VIEW_TARGET).addScaledVector(VIEW_DIR, d);
            camera.lookAt(VIEW_TARGET);
            camera.updateMatrixWorld();
            if (corners.every((c) => { probe.copy(c).project(camera); return Math.abs(probe.x) < 0.96 && Math.abs(probe.y) < 0.9; })) break;
        }
        controls.update();
    }
    fitCamera();

    const parts = buildScene(scene);

    // Plates: the moving unit, the OK stack and the reject pile
    const plateGeo = new THREE.BoxGeometry(PLATE.l, PLATE.h, PLATE.w);
    const edgeMat = new THREE.MeshStandardMaterial({ color: 0x7c8794, metalness: 0.9, roughness: 0.35 });
    const topMat = (map) => new THREE.MeshStandardMaterial({ map, metalness: 0.78, roughness: 0.36 });
    const cleanTop = topMat(plateTexture(null, 7));
    const defectTops = {};
    const defectTop = (d) => (defectTops[d] ||= topMat(plateTexture(d, 1000 + d.length * 17)));
    const plateMesh = (top) => {
        const m = new THREE.Mesh(plateGeo, [edgeMat, edgeMat, top, edgeMat, edgeMat, edgeMat]);
        m.castShadow = true; m.receiveShadow = true;
        return m;
    };
    const unit = new THREE.Group();
    const unitPlate = plateMesh(cleanTop);
    unit.add(unitPlate);
    const verdictGlowMat = new THREE.MeshBasicMaterial({ color: COLORS.ok, transparent: true, opacity: 0, depthWrite: false,
        blending: THREE.AdditiveBlending, toneMapped: false });
    const verdictGlow = new THREE.Mesh(new THREE.PlaneGeometry(PLATE.l + 0.5, PLATE.w + 0.45), verdictGlowMat);
    verdictGlow.rotation.x = -Math.PI / 2; verdictGlow.position.y = -PLATE.h / 2 - 0.004;
    unit.add(verdictGlow);
    const scanLine = box(0.03, 0.004, PLATE.w + 0.02, glowMaterial(COLORS.scan, 4), 0, PLATE.h / 2 + 0.003, 0, unit);
    scanLine.castShadow = false;
    scene.add(unit);

    const stack = [];
    for (let i = 0; i < PALLET_MAX; i++) {
        const m = plateMesh(cleanTop);
        m.position.set(PALLET.x + (i % 2 ? 0.02 : -0.02), PALLET.top + PLATE.h / 2 + i * (PLATE.h + 0.004), 0);
        m.visible = false; scene.add(m); stack.push(m);
    }
    const pile = [];
    const defectNames = ["scratches", "pitted_surface", "inclusion", "patches", "crazing", "rolled-in_scale"];
    const pileRand = rng(42);
    for (let i = 0; i < BIN_MAX; i++) {
        const m = plateMesh(defectTop(defectNames[i % defectNames.length]));
        m.position.set(BIN.x + (pileRand() - 0.5) * 0.35, BIN.floor + PLATE.h / 2 + i * 0.05, BIN.z + (pileRand() - 0.5) * 0.25);
        m.rotation.set((pileRand() - 0.5) * 0.25, (pileRand() - 0.5) * 0.6, (pileRand() - 0.5) * 0.2);
        m.visible = false; scene.add(m); pile.push(m);
    }

    // HUD overlay
    const hud = document.createElement("div");
    hud.className = "twin-hud";
    hud.innerHTML = `
        <div class="twin-chip twin-tl"><strong>LINE 01 · 3D DIGITAL TWIN</strong><span>Simulated feed · synced to /api/data</span></div>
        <div class="twin-chip twin-tr"><span class="twin-unit" id="twin-unit">U_0000</span><span class="twin-stage" id="twin-stage">Idle</span><span class="twin-progress"><i id="twin-bar"></i></span></div>
        <div class="twin-tag" id="twin-tag" hidden></div>
        <div class="twin-label" id="twin-cam">CAM_01 · QC SCAN</div>
        <div class="twin-label twin-label-ok" id="twin-ok">OK STACK · 0</div>
        <div class="twin-label twin-label-nok" id="twin-nok">REJECT · 0</div>
        <div class="twin-banner" id="twin-banner" hidden><strong></strong><span></span></div>`;
    container.appendChild(hud);
    const el = (id) => document.getElementById(id);
    document.getElementById("twin-loading")?.remove();

    // Post-processing: gentle bloom on lamps, lasers and verdict glow
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(rect.width, rect.height), 0.55, 0.5, 0.82);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());

    // ── Sync with the simulation ──
    // The clock is derived from the last server sample plus real elapsed time, so it stays correct
    // even when requestAnimationFrame pauses (background tab); `offset` smooths poll-to-poll jitter.
    const sim = { clock: 0, base: 0, baseAt: 0, offset: 0, running: false, mode: "PAUSED", cycle: 4, snap: null, synced: false };
    const targetAt = (t) => sim.base + (sim.running ? (t - sim.baseAt) / 1000 : 0);
    const statusByCycle = new Map();
    const counts = { ok: 0, nok: 0 };
    let lastCycle = -1, lastP = 0;

    async function poll() {
        try {
            const res = await fetch("/api/data", { cache: "no-store" });
            if (!res.ok) throw new Error(res.status);
            const d = await res.json();
            const t = performance.now();
            const before = targetAt(t);
            sim.snap = d;
            sim.mode = d.system_mode;
            sim.running = d.system_mode === "RUNNING";
            sim.cycle = d.cycle_seconds || 4;
            sim.base = d.sim_time; sim.baseAt = t;
            sim.offset += targetAt(t) - before;
            if (!sim.synced || Math.abs(sim.offset) > 0.6) sim.offset = 0;
            sim.clock = targetAt(t) - sim.offset;
            if (d.status_cycle >= 0 && d.current_unit_status !== "PENDING") {
                statusByCycle.set(d.status_cycle, { status: d.current_unit_status, defect: d.current_defect });
                if (statusByCycle.size > 16) statusByCycle.delete(statusByCycle.keys().next().value);
            }
            const localTotal = counts.ok + counts.nok;
            if (!sim.synced || d.total_units === 0 || Math.abs(d.total_units - localTotal) > 1) {
                counts.ok = d.ok_units; counts.nok = d.nok_units;
            }
            if (!sim.synced) { lastCycle = Math.floor(sim.clock / sim.cycle); lastP = (sim.clock % sim.cycle) / sim.cycle; }
            sim.synced = true;
        } catch (err) {
            sim.running = false;
        }
    }
    poll();
    setInterval(poll, POLL_MS);

    function land(cycle) {
        const s = statusByCycle.get(cycle);
        if (!s) return;
        if (s.status === "OK") counts.ok += 1; else counts.nok += 1;
    }

    // ── Per-frame update ──
    const tmp = new THREE.Vector3();
    function place(node, pos, dx = 0, dy = 0) {
        tmp.copy(pos).project(camera);
        const w = container.clientWidth, h = container.clientHeight;
        node.style.transform = `translate(-50%, -100%) translate(${(tmp.x * 0.5 + 0.5) * w + dx}px, ${(-tmp.y * 0.5 + 0.5) * h + dy}px)`;
        node.style.opacity = tmp.z < 1 ? "" : "0";
    }

    const labelPos = {
        cam: new THREE.Vector3(GANTRY_X, 3.5, 0),
        ok: new THREE.Vector3(PALLET.x, 1.35, 0),
        nok: new THREE.Vector3(BIN.x, 1.05, BIN.z + 0.3),
    };
    let last = performance.now();
    let blink = 0;

    function frame(now) {
        requestAnimationFrame(frame);
        const dt = Math.min(0.05, (now - last) / 1000);
        last = now;
        sim.offset *= Math.exp(-dt * 4);
        sim.clock = targetAt(now) - sim.offset;

        const cycle = Math.floor(sim.clock / sim.cycle);
        const p = (sim.clock % sim.cycle) / sim.cycle;
        if (sim.synced) {
            if (cycle === lastCycle && lastP < PHASE.land && p >= PHASE.land) land(cycle);
            else if (cycle > lastCycle && lastP < PHASE.land && lastCycle >= 0) land(lastCycle);
        }
        lastCycle = cycle; lastP = p;
        const info = statusByCycle.get(cycle);
        const status = info?.status ?? null;
        const failed = status === "FAIL";
        const started = (sim.snap?.total_units ?? 0) > 0 || sim.clock > 0;

        // Belt and rollers
        const speed = sim.running ? 16 / sim.cycle : 0;
        parts.beltTex.offset.x -= speed * dt;
        parts.rollers.forEach((r) => { r.rotation.y -= (speed * dt) / 0.16; });

        // Unit pose along its route
        unitPlate.material[2] = failed && info.defect ? defectTop(info.defect) : cleanTop;
        let x = xAt(p), y = PLATE_Y, z = 0, rx = 0, rz = 0;
        let pushZ = PADDLE_REST_Z;
        const approach = PHASE.pushStart - 0.05;
        if (failed && p >= approach && p < PHASE.pushStart) {
            pushZ = lerp(PADDLE_REST_Z, -PLATE.w / 2 - 0.03, ease((p - approach) / 0.05));
        }
        if (failed && p >= PHASE.pushStart) {
            x = DIVERTER_X;
            const push = ease(clamp01((p - PHASE.pushStart) / (PHASE.pushEnd - PHASE.pushStart)));
            const fall = ease(clamp01((p - PHASE.pushEnd) / (PHASE.fallEnd - PHASE.pushEnd)));
            z = lerp(0, 1.75, push);
            pushZ = z - PLATE.w / 2 - 0.03;
            if (p > PHASE.pushEnd) {
                pushZ = lerp(1.75 - PLATE.w / 2 - 0.03, PADDLE_REST_Z, ease(clamp01((p - PHASE.pushEnd) / (PHASE.retractEnd - PHASE.pushEnd))));
                const pileTop = BIN.floor + PLATE.h / 2 + Math.min(counts.nok, BIN_MAX) * 0.05;
                z = lerp(1.75, BIN.z, fall);
                y = lerp(PLATE_Y, pileTop, fall);
                rx = lerp(0, 0.35, Math.sin(fall * Math.PI)) + fall * 0.08;
            }
        } else if (!failed && p > PHASE.beltEnd) {
            const t = ease(clamp01((p - PHASE.beltEnd) / (PHASE.palletLand - PHASE.beltEnd)));
            const stackTop = PALLET.top + PLATE.h / 2 + (counts.ok % PALLET_MAX) * (PLATE.h + 0.004);
            x = lerp(BELT.end, PALLET.x, t);
            y = lerp(PLATE_Y, stackTop, t) + Math.sin(t * Math.PI) * 0.3;
            rz = -Math.sin(t * Math.PI) * 0.18;
        }
        unit.position.set(x, y, z);
        unit.rotation.set(rx, 0, rz);
        unit.visible = started && p < PHASE.land;
        // Paddle and rod live in the pusher group (world z = PUSHER_Z); the rod spans body face → paddle.
        parts.paddle.position.z = pushZ - PUSHER_Z;
        parts.rod.scale.z = Math.max(0.05, parts.paddle.position.z - ROD_BASE_Z);
        parts.rod.position.z = (parts.paddle.position.z + ROD_BASE_Z) / 2;

        // Scan effects
        const scanning = sim.running && p >= PHASE.scanStart && p <= PHASE.scanEnd;
        parts.sheetMat.opacity = scanning ? 0.22 : 0.07;
        scanLine.visible = scanning;
        scanLine.position.x = GANTRY_X - x;
        parts.ring.material.color.setHex(COLORS.scan).multiplyScalar(scanning ? 4 : 1.6);

        // Verdict
        const showVerdict = status && p >= PHASE.verdict && unit.visible;
        verdictGlowMat.color.setHex(failed ? COLORS.fail : COLORS.ok).multiplyScalar(2);
        verdictGlowMat.opacity = showVerdict ? (reducedMotion ? 0.55 : 0.45 + 0.15 * Math.sin(now / 180)) : 0;
        const tag = el("twin-tag");
        if (showVerdict) {
            const unitId = sim.snap?.current_unit_id ?? "";
            tag.hidden = false;
            tag.className = `twin-tag ${failed ? "is-fail" : "is-ok"}`;
            tag.innerHTML = failed
                ? `<strong>NOK</strong> ${info.defect.replace(/_/g, " ")}<span>${unitId}</span>`
                : `<strong>OK</strong> surface clean<span>${unitId}</span>`;
            place(tag, tmp.set(x, y + 0.45, z), 0, 0);
        } else {
            tag.hidden = true;
        }

        // Stacks
        const okShown = counts.ok === 0 ? 0 : ((counts.ok - 1) % PALLET_MAX) + 1;
        stack.forEach((m, i) => { m.visible = i < okShown; });
        const nokShown = Math.min(counts.nok, BIN_MAX);
        pile.forEach((m, i) => { m.visible = i < nokShown; });

        // Stack light
        blink += dt;
        const on = (name, lit) => {
            const lamp = parts.lamps[name];
            lamp.mat.color.copy(lamp.base).multiplyScalar(lit ? 3.2 : 0.18);
        };
        on("green", sim.mode === "RUNNING");
        on("amber", sim.mode === "PAUSED");
        on("red", sim.mode === "ESTOP" && (reducedMotion || Math.floor(blink * 2.5) % 2 === 0));

        // HUD
        el("twin-unit").textContent = sim.snap?.current_unit_id ?? "U_0000";
        el("twin-bar").style.width = `${(started ? p : 0) * 100}%`;
        el("twin-stage").textContent = !started ? "Idle"
            : p < 0.19 ? "Feeding" : p < PHASE.scanStart ? "Conveying" : p <= PHASE.scanEnd ? "Scanning"
            : failed ? (p < PHASE.pushStart - 0.05 ? "Conveying" : p < PHASE.fallEnd ? "Rejecting" : "Binned")
            : p < PHASE.beltEnd ? "Conveying" : "Stacking";
        el("twin-ok").textContent = `OK STACK · ${counts.ok}`;
        el("twin-nok").textContent = `REJECT · ${counts.nok}`;
        place(el("twin-cam"), labelPos.cam);
        place(el("twin-ok"), tmp.set(PALLET.x, PALLET.top + 0.6 + okShown * (PLATE.h + 0.004), 0));
        place(el("twin-nok"), labelPos.nok);
        const banner = el("twin-banner");
        if (sim.mode !== "RUNNING" && sim.synced) {
            banner.hidden = false;
            banner.className = `twin-banner ${sim.mode === "ESTOP" ? "is-estop" : ""}`;
            banner.querySelector("strong").textContent = sim.mode === "ESTOP" ? "EMERGENCY STOP" : `SYSTEM ${sim.mode}`;
            banner.querySelector("span").textContent = sim.mode === "ESTOP" ? "Line halted. Master Reset, then Start Cycle." : "Press Start Cycle to run the line.";
        } else {
            banner.hidden = true;
        }

        // Gentle idle camera drift until the user takes over
        if (!reducedMotion && !userMoved) {
            const t = now / 1000;
            controls.target.set(VIEW_TARGET.x + Math.sin(t * 0.15) * 0.3, VIEW_TARGET.y, VIEW_TARGET.z);
        }
        controls.update();
        composer.render();
    }

    let userMoved = false;
    controls.addEventListener("start", () => { userMoved = true; });

    new ResizeObserver(() => {
        const r = container.getBoundingClientRect();
        if (!r.width || !r.height) return;
        camera.aspect = r.width / r.height;
        camera.updateProjectionMatrix();
        renderer.setSize(r.width, r.height);
        composer.setSize(r.width, r.height);
        if (!userMoved) fitCamera();
    }).observe(container);

    window.lineTwinReady = true;
    requestAnimationFrame(frame);
}

init();
