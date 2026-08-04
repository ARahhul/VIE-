import { useEffect, useRef } from 'react'
import * as THREE from 'three'

// Animated "retro-futuristic grid cityscape" background: a pink-emissive
// grid plane with randomly-pulsing pillars, slow forward travel, subtle
// mouse parallax. Purely decorative (pointer-events disabled by the
// container), ported from the Stitch design export into a proper React
// component instead of an injected <script> tag.
function Background3D() {
  const containerRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const width = container.clientWidth || window.innerWidth
    const height = container.clientHeight || window.innerHeight

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.2))
    const spotLight = new THREE.SpotLight(0xff2d78, 2)
    spotLight.position.set(5, 10, 5)
    scene.add(spotLight)
    const pointLight = new THREE.PointLight(0xff2d78, 1, 50)
    pointLight.position.set(-5, -5, 5)
    scene.add(pointLight)

    const group = new THREE.Group()

    const gridHelper = new THREE.GridHelper(100, 40, 0xff2d78, 0x1a1a1a)
    gridHelper.position.y = -5
    group.add(gridHelper)

    const pillarGeo = new THREE.BoxGeometry(1, 1, 1)
    const pillars = []
    for (let i = 0; i < 30; i++) {
      const material = new THREE.MeshPhongMaterial({
        color: 0x111111,
        emissive: 0xff2d78,
        emissiveIntensity: Math.random() * 0.3,
        transparent: true,
        opacity: 0.9,
      })
      const pillar = new THREE.Mesh(pillarGeo, material)
      const x = (Math.random() - 0.5) * 40
      const z = (Math.random() - 0.5) * 40
      const h = 2 + Math.random() * 10
      pillar.position.set(x, -5 + h / 2, z)
      pillar.scale.set(1, h, 1)
      group.add(pillar)
      pillars.push({ mesh: pillar, speed: 0.01 + Math.random() * 0.02 })
    }

    scene.add(group)
    camera.position.set(0, 2, 15)
    camera.lookAt(0, 0, 0)

    let mouseX = 0
    let mouseY = 0
    const handleMouseMove = (e) => {
      mouseX = (e.clientX - window.innerWidth / 2) * 0.0005
      mouseY = (e.clientY - window.innerHeight / 2) * 0.0005
    }
    window.addEventListener('mousemove', handleMouseMove)

    let frameId
    const animate = () => {
      frameId = requestAnimationFrame(animate)

      group.position.z += 0.05
      if (group.position.z > 20) group.position.z = 0

      camera.position.x += (mouseX * 10 - camera.position.x) * 0.05
      camera.position.y += (-mouseY * 10 + 2 - camera.position.y) * 0.05
      camera.lookAt(0, 0, 0)

      pillars.forEach((p) => {
        p.mesh.material.emissiveIntensity = 0.1 + Math.sin(Date.now() * p.speed) * 0.2
      })

      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      const w = container.clientWidth || window.innerWidth
      const h = container.clientHeight || window.innerHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(frameId)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('resize', handleResize)
      renderer.dispose()
      pillarGeo.dispose()
      pillars.forEach((p) => p.mesh.material.dispose())
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  return <div ref={containerRef} className="background-3d" aria-hidden="true" />
}

export default Background3D
