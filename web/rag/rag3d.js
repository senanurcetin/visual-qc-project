// Three.js renderer for the RAG map. Imported lazily by rag.js; exposes the same
// interface as the 2D renderer: setQuery(sample), focus(index), destroy().
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const SCALE = 10;
const BASE_SIZE = 4.2;
const HIT_SIZE = 10;

const pointVertex = `
    attribute float size;
    attribute float alpha;
    attribute vec3 aColor;
    uniform float uPixelRatio;
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
        vColor = aColor;
        vAlpha = alpha;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * uPixelRatio * (40.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
    }`;

const pointFragment = `
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.5, 0.36, d));
    }`;

const glowFragment = `
    varying vec3 vColor;
    varying float vAlpha;
    uniform float uPulse;
    void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        float g = pow(1.0 - d * 2.0, 2.2);
        gl_FragColor = vec4(vColor, vAlpha * g * uPulse);
    }`;

export function createMap3D(frame, data, { colors, reducedMotion, onHover }) {
    const pts = data.map.points;
    const n = pts.length;
    const rect = frame.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(rect.width, rect.height);
    renderer.domElement.setAttribute("role", "img");
    renderer.domElement.setAttribute("aria-label", "3D map of passage embeddings coloured by topic, with the selected question and its retrieved passages highlighted");
    frame.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, rect.width / rect.height, 0.1, 200);
    camera.position.set(0, 6, 30);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enablePan = false;
    controls.minDistance = 8;
    controls.maxDistance = 60;
    controls.autoRotate = !reducedMotion;
    controls.autoRotateSpeed = 0.45;
    controls.addEventListener("start", () => { controls.autoRotate = false; });

    // Passage cloud
    const positions = new Float32Array(n * 3);
    const colorArr = new Float32Array(n * 3);
    const sizes = new Float32Array(n).fill(BASE_SIZE);
    const alphas = new Float32Array(n).fill(0.85);
    const tmp = new THREE.Color();
    pts.forEach((p, i) => {
        positions.set([p[0] * SCALE, p[1] * SCALE, p[2] * SCALE], i * 3);
        tmp.set(colors[p[5]]);
        colorArr.set([tmp.r, tmp.g, tmp.b], i * 3);
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aColor", new THREE.BufferAttribute(colorArr, 3));
    geometry.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
    geometry.setAttribute("alpha", new THREE.BufferAttribute(alphas, 1));
    const material = new THREE.ShaderMaterial({
        uniforms: { uPixelRatio: { value: pixelRatio } },
        vertexShader: pointVertex, fragmentShader: pointFragment,
        transparent: true, depthWrite: false,
    });
    const cloud = new THREE.Points(geometry, material);
    scene.add(cloud);

    // Glow layer for the query and the retrieved passages (additive).
    const glowGeometry = new THREE.BufferGeometry();
    const glowMaterial = new THREE.ShaderMaterial({
        uniforms: { uPixelRatio: { value: pixelRatio }, uPulse: { value: 1 } },
        vertexShader: pointVertex, fragmentShader: glowFragment,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const glow = new THREE.Points(glowGeometry, glowMaterial);
    scene.add(glow);

    // Query marker (drawn crisp on top of its glow).
    const queryGeometry = new THREE.BufferGeometry();
    queryGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3), 3));
    queryGeometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array([1, 1, 1]), 3));
    queryGeometry.setAttribute("size", new THREE.BufferAttribute(new Float32Array([11]), 1));
    queryGeometry.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array([1]), 1));
    const queryPoint = new THREE.Points(queryGeometry, material);
    queryPoint.visible = false;
    scene.add(queryPoint);

    // Lines from the question to each retrieved passage.
    const lineGeometry = new THREE.BufferGeometry();
    const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ color: 0xe6efff, transparent: true, opacity: 0.8 }));
    scene.add(lines);

    // Rank labels are HTML so they stay sharp and readable.
    const labels = [];
    const labelLayer = document.createElement("div");
    labelLayer.style.cssText = "position:absolute;inset:0;pointer-events:none;overflow:hidden";
    frame.appendChild(labelLayer);

    let sample = null, queryVec = new THREE.Vector3(), targets = [], growStart = 0, focused = null;
    const desiredTarget = new THREE.Vector3();
    const FOCUS_DISTANCE = 18;
    const offset = new THREE.Vector3();

    function setQuery(s) {
        sample = s;
        focused = null;
        const retrieved = new Set(s.retrieved.map((r) => r.index));
        for (let i = 0; i < n; i++) {
            sizes[i] = retrieved.has(i) ? HIT_SIZE : BASE_SIZE;
            alphas[i] = retrieved.has(i) ? 1 : 0.32;
        }
        geometry.attributes.size.needsUpdate = true;
        geometry.attributes.alpha.needsUpdate = true;

        queryVec.set(s.query_3d[0] * SCALE, s.query_3d[1] * SCALE, s.query_3d[2] * SCALE);
        queryGeometry.attributes.position.array.set(queryVec.toArray());
        queryGeometry.attributes.position.needsUpdate = true;
        queryPoint.visible = true;
        targets = s.retrieved.map((r) => new THREE.Vector3().fromArray(positions, r.index * 3));

        const gPos = [queryVec.x, queryVec.y, queryVec.z], gCol = [1, 1, 1], gSize = [34], gAlpha = [0.55];
        s.retrieved.forEach((r, k) => {
            gPos.push(...targets[k].toArray());
            tmp.set(colors[pts[r.index][5]]);
            gCol.push(tmp.r, tmp.g, tmp.b); gSize.push(30); gAlpha.push(0.7);
        });
        glowGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(gPos), 3));
        glowGeometry.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(gCol), 3));
        glowGeometry.setAttribute("size", new THREE.BufferAttribute(new Float32Array(gSize), 1));
        glowGeometry.setAttribute("alpha", new THREE.BufferAttribute(new Float32Array(gAlpha), 1));

        lineGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(targets.length * 6), 3));
        growStart = performance.now();
        updateLines(reducedMotion ? 1 : 0);

        labels.forEach((el) => el.remove());
        labels.length = 0;
        s.retrieved.forEach((r) => {
            const el = document.createElement("span");
            el.textContent = r.rank;
            el.style.cssText = "position:absolute;font:600 11px 'IBM Plex Mono',monospace;color:#07111f;background:#e6efff;border-radius:999px;padding:0 5px;transform:translate(8px,-18px)";
            labelLayer.appendChild(el);
            labels.push(el);
        });
        desiredTarget.copy(queryVec).multiplyScalar(0.6);
    }

    function updateLines(t) {
        const e = 1 - Math.pow(1 - Math.min(1, t), 3);
        const arr = lineGeometry.attributes.position.array;
        targets.forEach((p, k) => {
            arr.set(queryVec.toArray(), k * 6);
            arr.set(queryVec.clone().lerp(p, e).toArray(), k * 6 + 3);
        });
        lineGeometry.attributes.position.needsUpdate = true;
    }

    function focus(i) {
        focused = i;
        if (!sample) return;
        const retrieved = new Set(sample.retrieved.map((r) => r.index));
        for (const idx of retrieved) sizes[idx] = idx === i ? HIT_SIZE * 1.6 : HIT_SIZE;
        geometry.attributes.size.needsUpdate = true;
    }

    // Hover picking
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.22;
    const pointer = new THREE.Vector2();
    let hoverEvent = null;
    const onMove = (e) => { hoverEvent = e; };
    const onLeave = () => { hoverEvent = null; onHover(null); };
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("pointerleave", onLeave);

    function pick() {
        if (!hoverEvent) return;
        const r = renderer.domElement.getBoundingClientRect();
        const x = hoverEvent.clientX - r.left, y = hoverEvent.clientY - r.top;
        pointer.set((x / r.width) * 2 - 1, -(y / r.height) * 2 + 1);
        raycaster.setFromCamera(pointer, camera);
        const hits = raycaster.intersectObject(cloud);
        onHover(hits.length ? hits[0].index : null, x, y);
        hoverEvent = null;
    }

    const projected = new THREE.Vector3();
    function placeLabels() {
        const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
        labels.forEach((el, k) => {
            projected.copy(targets[k]).project(camera);
            el.style.left = `${(projected.x * 0.5 + 0.5) * w}px`;
            el.style.top = `${(-projected.y * 0.5 + 0.5) * h}px`;
            el.style.display = projected.z < 1 ? "" : "none";
        });
    }

    let raf = 0, onScreen = true;
    function frameLoop(now) {
        raf = requestAnimationFrame(frameLoop);
        if (!onScreen || document.hidden) return;
        if (sample && !reducedMotion) {
            const t = (now - growStart) / 800;
            if (t <= 1.05) updateLines(t);
            glowMaterial.uniforms.uPulse.value = 0.8 + 0.2 * Math.sin(now / 420);
        }
        controls.target.lerp(desiredTarget, reducedMotion ? 1 : 0.04);
        if (sample && now - growStart < 1500) {
            // Ease the camera toward the question once per selection; afterwards the user owns the zoom.
            offset.copy(camera.position).sub(controls.target);
            const dist = offset.length();
            offset.setLength(dist + (FOCUS_DISTANCE - dist) * (reducedMotion ? 1 : 0.05));
            camera.position.copy(controls.target).add(offset);
        }
        controls.update();
        pick();
        placeLabels();
        renderer.render(scene, camera);
    }
    raf = requestAnimationFrame(frameLoop);

    const visibility = new IntersectionObserver(([entry]) => { onScreen = entry.isIntersecting; });
    visibility.observe(frame);
    const resize = new ResizeObserver(() => {
        const r = frame.getBoundingClientRect();
        if (!r.width || !r.height) return;
        camera.aspect = r.width / r.height;
        camera.updateProjectionMatrix();
        renderer.setSize(r.width, r.height);
    });
    resize.observe(frame);

    return {
        setQuery,
        focus,
        destroy() {
            cancelAnimationFrame(raf);
            visibility.disconnect();
            resize.disconnect();
            controls.dispose();
            [geometry, glowGeometry, queryGeometry, lineGeometry].forEach((g) => g.dispose());
            [material, glowMaterial, lines.material].forEach((m) => m.dispose());
            renderer.dispose();
            renderer.domElement.remove();
            labelLayer.remove();
            onHover(null);
        },
    };
}
