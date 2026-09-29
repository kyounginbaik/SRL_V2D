"""Exercise the actual generated site, including software WebGL and video seeks."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--browser', required=True)
    parser.add_argument('--output', type=Path, default=Path('.ablation_lab/browser_test'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.browser, headless=True,
            args=['--no-sandbox', '--disable-dev-shm-usage', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1512, 'height': 1200}, device_scale_factor=1)
        errors = []; page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(args.url + '/?episode=41', wait_until='networkidle', timeout=120000)
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden", timeout=120000)
        page.screenshot(path=str(args.output / 'desktop.png'), full_page=True)
        assert page.locator('#pane0 .metric strong').first.inner_text().startswith('40.2')
        assert page.locator('#pane1 .metric strong').first.inner_text().startswith('137.6')
        page.locator('[data-preset="crop"]').click()
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden")
        assert page.locator('#pane0 .metric strong').nth(1).inner_text().startswith('12.53')
        assert page.locator('#pane1 .metric strong').nth(1).inner_text().startswith('0.88')
        page.screenshot(path=str(args.output / 'detector_crop.png'), full_page=True)
        page.locator('#cropProbeButton').click()
        assert '12.53 m' in page.locator('#cropProbeBody').inner_text()
        assert '0.62 m' in page.locator('#cropProbeBody').inner_text()
        page.wait_for_function("[...document.querySelectorAll('#cropProbeBody img')].every(i=>i.complete&&i.naturalWidth>0)")
        page.screenshot(path=str(args.output / 'crop_probe.png'), full_page=True)
        page.locator('#probe .close').click()
        page.locator('#worst').click()
        assert page.locator('#timeline').input_value() == '244'
        page.locator('[data-preset="depth"]').click()
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden")
        assert 'Depth-aligned observations' in page.locator('#pane1 .paneTitle').inner_text()
        page.locator('select[data-key="hand"]').select_option('hawor')
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden")
        assert page.locator('select[data-key="stage"]').input_value() == 'filled'
        assert page.locator('select[data-key="gravity"]').input_value() == 'on'
        assert page.locator('select[data-key="refine"]').input_value() == 'off'
        assert 'HaWoR' in page.locator('#pane1 .paneTitle').inner_text()
        assert page.locator('#pane1 .metric strong').nth(1).inner_text().startswith('0.86')
        page.screenshot(path=str(args.output / 'hawor.png'), full_page=True)
        page.locator('[data-preset="refine"]').click()
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden")
        page.locator('#pane0 [data-view="ego"]').click()
        page.wait_for_function("document.querySelector('#pane0 video').readyState>=2")
        page.locator('#timeline').fill('120');page.locator('#timeline').dispatch_event('input')
        page.wait_for_timeout(1200)
        timing=page.locator('#pane0 video').evaluate('(v)=>({time:v.currentTime,duration:v.duration})')
        assert timing['time'] > 0
        page.locator('#pane0 [data-view="retarget"]').click()
        assert 'INDEPENDENT OF A/B' in page.locator('#pane0 .viewLabel').inner_text()
        page.wait_for_function("document.querySelector('#pane0 video').readyState>=2")
        page.locator('#pane0 [data-view="lens"]').click()
        page.wait_for_function("document.querySelector('#pane0 video').readyState>=2")
        page.locator('#pane0 [data-view="depth"]').click()
        page.wait_for_function("document.querySelector('#pane0 .stageImage').complete && document.querySelector('#pane0 .stageImage').naturalWidth > 0")
        page.locator('#episode').select_option('episode_000000')
        page.wait_for_function("document.querySelector('#episodeNotice').textContent.includes('Incomplete reconstruction')")
        page.locator('#pane0 .empty').wait_for(state='visible', timeout=10000)
        page.locator('#episode').select_option('episode_000031')
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden")
        page.locator('#pane0 [data-view="world"]').click()
        page.locator('[data-preset="clean"]').click()
        page.wait_for_function("document.querySelector('#pane1 .empty').hidden")
        page.screenshot(path=str(args.output / 'cleanup.png'), full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(500)
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
        page.screenshot(path=str(args.output / 'mobile.png'), full_page=True)
        response=page.request.get(args.url+'/media/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_000041.mp4',headers={'Range':'bytes=0-99'})
        assert response.status==206 and len(response.body())==100
        assert not errors, errors
        (args.output/'results.json').write_text(json.dumps({'passed':True,'page_errors':errors,'overlay_timing':timing,'checks':['actual metrics','detector-crop experiment','outlier navigation','stage ablation','HaWoR experiment','video seek','IK provenance and playback','undistortion playback','metric depth preview','incomplete run','episode switch','mobile layout','HTTP byte ranges']},indent=2))
        browser.close()
        print('Browser checks passed')


if __name__ == '__main__':
    main()
