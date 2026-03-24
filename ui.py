import os
import json
import logging
import subprocess
import sys
from pathlib import Path
from threading import Thread
import gradio as gr

from phone_agent import PhoneAgent

# Platform detection
IS_WINDOWS = sys.platform.startswith('win')


class UILogHandler(logging.Handler):
    """Custom logging handler that stores logs for UI display."""
    
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        if len(self.logs) > 200:
            self.logs = self.logs[-200:]


# Global state
current_screenshot = None
log_handler = None
is_running = False
agent = None
current_config = None


def load_config(config_path="config.json"):
    """Load configuration from file."""
    if not os.path.exists(config_path):
        return get_default_config()
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        default = get_default_config()
        for key, value in default.items():
            if key not in config:
                config[key] = value
        return config
    except json.JSONDecodeError:
        return get_default_config()


def get_default_config():
    """Get default configuration."""
    return {
        "device_id": None,
        "screen_width": 1080,
        "screen_height": 2340,
        "screenshot_dir": "./screenshots",
        "max_retries": 3,
        "max_cycles": 10,
        "loop_detection_threshold": 3,
        "model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "use_flash_attention": False,
        "temperature": 0.1,
        "max_tokens": 512,
        "step_delay": 1.5,
        "enable_visual_debug": False
    }


def save_config(config, config_path="config.json"):
    """Save configuration to file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Failed to save config: {e}")
        return False


def setup_logging():
    """Configure logging for the UI."""
    global log_handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    log_handler = UILogHandler()
    log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    root_logger.addHandler(log_handler)
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler("phone_agent_ui.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler with UTF-8 encoding (platform-aware)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Try to set UTF-8 encoding on Windows
    if IS_WINDOWS and hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except (AttributeError, OSError):
            # Fallback for incompatible console or older Python
            logging.warning("Could not set UTF-8 encoding for console output")
    
    root_logger.addHandler(console_handler)


def detect_device_resolution():
    """Try to detect connected device resolution via ADB."""
    try:
        result = subprocess.run(
            ["adb", "shell", "wm", "size"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0 and "Physical size:" in result.stdout:
            size_str = result.stdout.split("Physical size:")[1].strip()
            width, height = map(int, size_str.split('x'))
            # Use ASCII-safe characters for cross-platform compatibility
            return width, height, f"[OK] Detected: {width} x {height}"
        else:
            return None, None, "[WARN] No device detected"
            
    except Exception as e:
        return None, None, f"[ERROR] {str(e)}"


def execute_task_thread(task, max_cycles, config):
    """Run task in background thread."""
    global current_screenshot, is_running, agent
    
    if log_handler:
        log_handler.logs.clear()
    
    is_running = True
    
    try:
        logging.info(f"Starting task: '{task}'")
        
        # Only create agent if it doesn't exist
        if agent is None:
            logging.info("=" * 60)
            logging.info("FIRST TIME SETUP - PLEASE WAIT")
            logging.info("=" * 60)
            logging.info("Initializing Phone Agent...")
            logging.info("This includes downloading and loading the AI model (~17GB)")
            logging.info("First-time setup can take 5-15 minutes depending on:")
            logging.info("  - Your internet speed (for download)")
            logging.info("  - Your hardware (for loading into memory)")
            logging.info("")
            logging.info("Please do not close this window!")
            logging.info("Progress will be shown below...")
            logging.info("=" * 60)
            
            # Set flag to use UI logging instead of creating new log files
            config['use_ui_logging'] = True
            
            agent = PhoneAgent(config)
            
            logging.info("=" * 60)
            logging.info("SETUP COMPLETE - Agent is ready!")
            logging.info("=" * 60)
        else:
            logging.info("Reusing existing agent...")
            # Reset context for new task
            from datetime import datetime
            agent.context['previous_actions'] = []
            agent.context['task_request'] = task
            agent.context['session_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent.context['screenshots'] = []
        
        # Monkey-patch to capture screenshots
        original_capture = agent.capture_screenshot
        def capture_with_tracking():
            path = original_capture()
            global current_screenshot
            current_screenshot = path
            return path
        
        agent.capture_screenshot = capture_with_tracking
        
        # Execute task - use max_cycles from UI input, or fall back to agent config
        result = agent.execute_task(task, max_cycles=int(max_cycles) if max_cycles else None)
        
        if result['success']:
            logging.info(f"✓ Task completed in {result['cycles']} cycles")
        else:
            logging.info(f"⚠️ Task incomplete after {result['cycles']} cycles")
            
    except KeyboardInterrupt:
        logging.info("Task interrupted by user")
    except Exception as e:
        logging.error(f"Task execution error: {e}", exc_info=True)
    finally:
        is_running = False


def start_task(task, max_cycles, config_json):
    """Start a task execution."""
    global is_running, current_config
    
    if is_running:
        return (
            "⚠️ A task is already running",
            None,
            "No logs yet...",  # ← Add initial log state
            gr.update(active=False)
        )
    
    if not task.strip():
        return (
            "✗ Please enter a task",
            None,
            "No logs yet...",  # ← Add initial log state
            gr.update(active=False)
        )
    
    try:
        config = json.loads(config_json)
        current_config = config
    except json.JSONDecodeError as e:
        return (
            f"✗ Invalid config JSON: {e}",
            None,
            f"Configuration error: {e}",  # ← Show error in logs
            gr.update(active=False)
        )
    
    try:
        max_cycles = int(max_cycles)
        if max_cycles < 1:
            max_cycles = 15
    except ValueError:
        max_cycles = 15
    
    thread = Thread(target=execute_task_thread, args=(task, max_cycles, config))
    thread.daemon = True
    thread.start()
    
    return (
        "✓ Task started...",
        None,
        "Task starting...\n",  # ← Initialize log display
        gr.update(active=True)  # ← Activate timer immediately
    )


def update_ui():
    """Update UI with latest screenshot and logs."""
    global current_screenshot, log_handler, is_running
    
    screenshot = None
    if current_screenshot and os.path.exists(current_screenshot):
        screenshot = current_screenshot
    
    # Get logs from handler
    logs = "\n".join(log_handler.logs) if log_handler and log_handler.logs else "No logs yet..."
    
    # Keep timer active while task is running
    timer_state = gr.update(active=is_running)
    
    return (screenshot, logs, timer_state)


def stop_task():
    """Stop the currently running task."""
    global is_running
    if is_running:
        logging.warning("Task stop requested by user")
        is_running = False
        return "⚠️ Stopping task..."
    return "No task running"


def apply_settings(screen_width, screen_height, temp, max_tok, step_delay, use_fa2, visual_debug):
    """Apply settings changes to config."""
    global current_config
    
    try:
        config = current_config or load_config()
        
        config['screen_width'] = int(screen_width)
        config['screen_height'] = int(screen_height)
        config['temperature'] = float(temp)
        config['max_tokens'] = int(max_tok)
        config['step_delay'] = float(step_delay)
        config['use_flash_attention'] = use_fa2
        config['enable_visual_debug'] = visual_debug
        
        if save_config(config):
            current_config = config
            return "✓ Settings saved", json.dumps(config, indent=2)
        else:
            return "✗ Failed to save settings", json.dumps(config, indent=2)
            
    except ValueError as e:
        return f"✗ Invalid value: {e}", json.dumps(current_config or {}, indent=2)


def auto_detect_resolution():
    """Auto-detect device resolution."""
    width, height, message = detect_device_resolution()
    
    if width and height:
        return width, height, message
    else:
        return 1080, 2340, message


def clear_logs_fn():
    """Clear the log display."""
    if log_handler:
        log_handler.logs.clear()
    return ""


def create_ui():
    """Create the Gradio interface."""
    global current_config
    current_config = load_config()
    
    Path(current_config['screenshot_dir']).mkdir(parents=True, exist_ok=True)
    
    with gr.Blocks(title="Phone Agent Control Panel", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", font=["Inter", "system-ui", "sans-serif"])) as demo:
        gr.Markdown("# 📱 Phone Agent Control Panel")
        gr.Markdown("*Intelligent visual automation powered by Qwen3-VL AI*")
        
        with gr.Tabs():
            with gr.Tab("🎯 Control Center"):
                with gr.Row():
                    with gr.Column(scale=2):
                        task_input = gr.Textbox(
                            label="📝 Task Instructions",
                            placeholder="e.g., 'Open Chrome and search for weather in New York'",
                            lines=3
                        )
                        
                        with gr.Row():
                            max_cycles = gr.Number(
                                label="🔄 Max Cycles",
                                value=current_config.get('max_cycles', 10),
                                minimum=1,
                                maximum=50
                            )
                            start_btn = gr.Button("🚀 Launch Task", variant="primary", scale=2)
                            stop_btn = gr.Button("🛑 Stop Task", variant="stop", scale=1)
                        
                        status_text = gr.Textbox(label="📊 Task Status", lines=2, interactive=False)
                    
                    with gr.Column(scale=3):
                        image_output = gr.Image(
                            label="📺 Live Device Screen",
                            type="filepath",
                            height=600
                        )
                
                log_output = gr.Textbox(
                    label="📜 Execution Log",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                    value="🟢 Ready to start. Click 'Launch Task' above to begin."
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh View", variant="secondary")
                    clear_logs_btn = gr.Button("🧹 Clear Logs", variant="secondary")
            
            with gr.Tab("⚙️ Settings"):
                gr.Markdown("### 📐 Device Configuration")
                
                with gr.Row():
                    with gr.Column():
                        detect_btn = gr.Button("🔍 Auto-Detect Resolution", variant="secondary")
                        detect_status = gr.Textbox(label="Detection Result", interactive=False)
                    
                    with gr.Column():
                        screen_width = gr.Number(
                            label="📏 Screen Width (px)",
                            value=current_config['screen_width']
                        )
                        screen_height = gr.Number(
                            label="📏 Screen Height (px)",
                            value=current_config['screen_height']
                        )
                
                gr.Markdown("### 🎮 Task Execution Settings")
                
                with gr.Row():
                    max_retries = gr.Number(
                        label="🔁 Max Retries (per action)",
                        value=current_config.get('max_retries', 3),
                        minimum=1,
                        maximum=10
                    )
                    default_max_cycles = gr.Number(
                        label="🔄 Default Max Cycles",
                        value=current_config.get('max_cycles', 10),
                        minimum=1,
                        maximum=50
                    )
                
                with gr.Row():
                    loop_threshold = gr.Number(
                        label="🔁 Loop Detection Threshold",
                        value=current_config.get('loop_detection_threshold', 3),
                        minimum=2,
                        maximum=5
                    )
                
                gr.Markdown("### 🤖 AI Model Parameters")
                
                with gr.Row():
                    temperature = gr.Slider(
                        label="🌡️ Temperature (creativity)",
                        minimum=0.0,
                        maximum=1.0,
                        value=current_config['temperature'],
                        step=0.05
                    )
                    max_tokens = gr.Number(
                        label="📝 Max Tokens",
                        value=current_config['max_tokens'],
                        minimum=128,
                        maximum=2048
                    )
                
                with gr.Row():
                    step_delay = gr.Slider(
                        label="⏱️ Step Delay (seconds)",
                        minimum=0.5,
                        maximum=5.0,
                        value=current_config['step_delay'],
                        step=0.1
                    )
                
                gr.Markdown("### 🔧 Advanced Options")
                
                with gr.Row():
                    use_flash_attn = gr.Checkbox(
                        label="⚡ Enable Flash Attention 2",
                        value=current_config.get('use_flash_attention', False)
                    )
                    visual_debug = gr.Checkbox(
                        label="🐛 Enable Visual Debugging",
                        value=current_config.get('enable_visual_debug', False)
                    )
                
                apply_btn = gr.Button("💾 Save Configuration", variant="primary")
                settings_status = gr.Textbox(label="💬 Save Status", interactive=False)
                
                gr.Markdown("### 📋 Configuration JSON")
                config_editor = gr.Code(
                    label="Raw Configuration",
                    language="json",
                    value=json.dumps(current_config, indent=2),
                    lines=15
                )
            
            with gr.Tab("📖 Documentation"):
                gr.Markdown("""
## 🚀 Quick Start Guide

1. **📱 Connect Device**: Ensure USB debugging is enabled and device is connected via ADB
2. **⚙️ Configure Resolution**: Navigate to Settings tab and use auto-detect
3. **▶️ Execute Task**: Enter your task description and click Launch Task

## 💡 Example Tasks

- 📧 "Open Gmail and compose a new email"
- 🌤️ "Search for weather forecast in Tokyo"
- 📷 "Open Camera and switch to video mode"
- ⚙️ "Navigate to Settings and enable Bluetooth"
- 🎵 "Open Spotify and play my liked songs"

## 🔧 Troubleshooting

- **❌ Incorrect Touch**: Verify screen resolution in Settings tab matches your device
- **🔌 Device Not Found**: Run `adb devices` in terminal to verify connection
- **⚠️ Task Errors**: Check the Execution Log for detailed error messages
- **🐌 Slow Performance**: Reduce max cycles or increase step delay in Settings

## 📊 Understanding Logs

- 🟢 **Green messages**: Successful operations
- 🟡 **Yellow warnings**: Non-critical issues
- 🔴 **Red errors**: Failed operations requiring attention
                """)
        
        # Timer with faster update interval for better log responsiveness
        timer = gr.Timer(value=1, active=False)
        
        # Event handlers
        start_btn.click(
            fn=start_task,
            inputs=[task_input, max_cycles, config_editor],
            outputs=[status_text, image_output, log_output, timer]  # ← Added log_output
        )
        
        stop_btn.click(
            fn=stop_task,
            outputs=status_text
        )
        
        timer.tick(
            fn=update_ui,
            outputs=[image_output, log_output, timer]
        )
        
        refresh_btn.click(
            fn=update_ui,
            outputs=[image_output, log_output, timer]
        )
        
        clear_logs_btn.click(
            fn=clear_logs_fn,
            outputs=log_output
        )
        
        detect_btn.click(
            fn=auto_detect_resolution,
            outputs=[screen_width, screen_height, detect_status]
        )
        
        apply_btn.click(
            fn=lambda sw, sh, temp, max_tok, step, use_fa2, vis_dbg, max_ret, max_cyc, loop_thresh: apply_settings_extended(
                sw, sh, temp, max_tok, step, use_fa2, vis_dbg, max_ret, max_cyc, loop_thresh
            ),
            inputs=[
                screen_width,
                screen_height,
                temperature,
                max_tokens,
                step_delay,
                use_flash_attn,
                visual_debug,
                max_retries,
                default_max_cycles,
                loop_threshold
            ],
            outputs=[settings_status, config_editor]
        )
    
    return demo

def apply_settings_extended(screen_width, screen_height, temp, max_tok, step_delay, use_fa2, visual_debug, max_retries, max_cycles, loop_threshold):
    """Apply settings changes to config including retry/cycle limits."""
    global current_config
    
    try:
        config = current_config or load_config()
        
        config['screen_width'] = int(screen_width)
        config['screen_height'] = int(screen_height)
        config['temperature'] = float(temp)
        config['max_tokens'] = int(max_tok)
        config['step_delay'] = float(step_delay)
        config['use_flash_attention'] = use_fa2
        config['enable_visual_debug'] = visual_debug
        config['max_retries'] = int(max_retries)
        config['max_cycles'] = int(max_cycles)
        config['loop_detection_threshold'] = int(loop_threshold)
        
        if save_config(config):
            current_config = config
            return "✓ Settings saved", json.dumps(config, indent=2)
        else:
            return "✗ Failed to save settings", json.dumps(config, indent=2)
            
    except ValueError as e:
        return f"✗ Invalid value: {e}", json.dumps(current_config or {}, indent=2)


def main():
    """Main entry point for the UI."""
    print("Phone Agent UI Starting...")
    print("Setting up logging...")
    setup_logging()
    
    print("Creating interface...")
    demo = create_ui()
    
    print("Starting server on http://localhost:7860")
    print("Press Ctrl+C to stop")
    
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()