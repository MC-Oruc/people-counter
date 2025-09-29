from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path

from .cli import main as cli_main
from .config import ensure_default_configs, DEFAULT_APP_CFG, DEFAULT_LINE_CFG


CONFIG_DIR = Path.cwd() / "config"
APP_YAML = DEFAULT_APP_CFG
LINE_YAML = DEFAULT_LINE_CFG


def _ensure_configs() -> None:
	ensure_default_configs()


class _ProcManager:
	def __init__(self) -> None:
		self.procs: dict[str, subprocess.Popen] = {}

	def is_running(self, key: str) -> bool:
		p = self.procs.get(key)
		return p is not None and (p.poll() is None)

	def start(self, key: str, args: list[str]) -> None:
		if self.is_running(key):
			print(f"[{key}] zaten çalışıyor.")
			return
		print(f"[{key}] başlatılıyor: {' '.join(args)}")
		self.procs[key] = subprocess.Popen(args)
		time.sleep(0.2)
		print(f"[{key}] durum: {'çalışıyor' if self.is_running(key) else 'bitti'}")

	def stop(self, key: str, timeout: float = 2.0) -> None:
		p = self.procs.get(key)
		if p is None or p.poll() is not None:
			print(f"[{key}] çalışmıyor.")
			return
		print(f"[{key}] durduruluyor...")
		p.terminate()
		try:
			p.wait(timeout=timeout)
		except Exception:
			print(f"[{key}] sonlandırılamadı, zorla kapatılıyor...")
			p.kill()
		print(f"[{key}] durdu.")

	def stop_all(self) -> None:
		for k in list(self.procs.keys()):
			self.stop(k)


def _print_menu(pm: _ProcManager) -> None:
	print()
	print("=== People Counter Launcher ===")
	print(f"Config: {APP_YAML if APP_YAML.exists() else '(yok)'}")
	print(f"Line:   {LINE_YAML if LINE_YAML.exists() else '(yok)'}")
	print("-------------------------------")
	print(f"1) Headless Başlat   [{'RUNNING' if pm.is_running('headless') else 'READY'}]")
	print(f"2) Headless Durdur   [{'READY' if not pm.is_running('headless') else 'STOP'}]")
	print(f"3) UI Başlat         [{'RUNNING' if pm.is_running('ui') else 'READY'}]")
	print(f"4) UI Durdur         [{'READY' if not pm.is_running('ui') else 'STOP'}]")
	print("5) Çizgiyi Düzenle ve Kaydet")
	print("6) Config Dosyalarını Oluştur")
	print("7) Çıkış")
	print()


def launcher_main() -> int:
	_ensure_configs()
	pm = _ProcManager()
	py = sys.executable
	while True:
		try:
			_print_menu(pm)
			choice = input("Seçim: ").strip()
			if choice == '1':
				args = [py, '-m', 'people_counter', 'run', '--', '--config', str(APP_YAML)]
				pm.start('headless', args)
			elif choice == '2':
				pm.stop('headless')
			elif choice == '3':
				args = [py, '-m', 'people_counter', 'ui']
				pm.start('ui', args)
			elif choice == '4':
				pm.stop('ui')
			elif choice == '5':
				args = [py, '-m', 'people_counter', 'edit-line', '--output', str(LINE_YAML)]
				subprocess.call(args)
			elif choice == '6':
				_ensure_configs()
				print("Config dosyaları hazır.")
			elif choice == '7' or choice.lower() in ('q', 'quit', 'exit'):
				print("Çıkılıyor...")
				pm.stop_all()
				break
			else:
				print("Geçersiz seçim.")
		except KeyboardInterrupt:
			print("\nCtrl+C algılandı. Çıkılıyor...")
			pm.stop_all()
			break
		except Exception as e:
			print(f"Hata: {e}")
	return 0


def main() -> int:
	# If there are CLI args, delegate to the existing CLI
	if len(sys.argv) > 1:
		return cli_main()
	# Otherwise, launch the interactive menu (former top-level launcher)
	return launcher_main()


if __name__ == "__main__":
	raise SystemExit(main())
