import os
import json
import time
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import Counter

from qwen_vl_agent import QwenVLAgent
from claude_api_agent import ClaudeAPIAgent
from device_backends import create_backend

# Platform detection
IS_WINDOWS = sys.platform.startswith('win')
IS_LINUX = sys.platform.startswith('linux')


class PhoneAgent:
    """
    Phone automation agent using Qwen3-VL for visual understanding and ADB for control.
    
    This agent:
    - Captures screenshots from Android devices via ADB
    - Uses Qwen3-VL to analyze screens and determine actions
    - Executes actions through ADB commands
    - Tracks context and action history
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the phone agent.
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        default_config = {
            'platform': 'android',  # 'android' or 'ios'
            'device_id': None,  # Auto-detect first device if None
            'screen_width': 1080,  # Must match your device
            'screen_height': 2340,  # Must match your device
            'screenshot_dir': './screenshots',
            'max_retries': 3,  # Max retries for FAILED actions within a cycle
            'max_cycles': 15,  # Max total cycles before giving up on task
            'model_name': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
            'use_flash_attention': False,
            'temperature': 0.1,
            'max_tokens': 512,
            'step_delay': 1.5,  # Seconds to wait after each action
            'enable_visual_debug': False,  # Save annotated screenshots
            'loop_detection_threshold': 3,  # Detect repeated actions after N occurrences
            'use_ui_logging': False,  # Set to True when running from UI
        }
        
        self.config = default_config
        if config:
            self.config.update(config)
        
        # Session context
        self.context = {
            'previous_actions': [],
            'current_app': "Home",
            'task_request': "",
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'screenshots': [],
            'failed_attempts': [],  # Track what didn't work
            'last_screen_hash': None,  # Detect if screen hasn't changed
        }
        
        # Setup logging only if not using UI logging
        if not self.config.get('use_ui_logging', False):
            self._setup_logging()
        else:
            # Just log the session start using existing handlers
            logging.info(f"Session started: {self.context['session_id']}")
            logging.info(f"Platform: {sys.platform}")
        
        # Initialize directories
        self._setup_directories()
        
        # Initialize device backend and connect
        self.device = create_backend(self.config.get('platform', 'android'))
        resolved_id = self.device.connect(self.config.get('device_id'))
        self.config['device_id'] = resolved_id
        self._sync_screen_resolution()
        
        # Initialize vision-language agent (API or local)
        if self.config.get('use_claude_api', False):
            logging.info("Initializing Claude API agent...")
            self.vl_agent = ClaudeAPIAgent(
                model=self.config.get('claude_model', 'claude-opus-4-6'),
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                api_key=self.config.get('anthropic_api_key') or None,
            )
        else:
            logging.info("Initializing Qwen3-VL agent...")
            self.vl_agent = QwenVLAgent(
                use_flash_attention=self.config.get('use_flash_attention', False),
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
            )
        logging.info("Phone agent ready")
    
    def _setup_logging(self):
        """Configure logging for this session."""
        log_file = f"phone_agent_{self.context['session_id']}.log"
        
        # Use UTF-8 encoding for all platforms
        handlers = [
            logging.FileHandler(log_file, encoding='utf-8')
        ]
        
        # Add console handler with proper encoding
        console_handler = logging.StreamHandler()
        if IS_WINDOWS and hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except (AttributeError, OSError):
                # Fallback for older Python or incompatible console
                pass
        handlers.append(console_handler)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )
        logging.info(f"Session started: {self.context['session_id']}")
        logging.info(f"Platform: {sys.platform}")
    
    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config['screenshot_dir']).mkdir(parents=True, exist_ok=True)
        logging.info(f"Screenshots directory: {self.config['screenshot_dir']}")
    
    def _sync_screen_resolution(self):
        """Read actual resolution from device and update config if it differs."""
        try:
            width, height = self.device.get_screen_resolution()
            if width != self.config['screen_width'] or height != self.config['screen_height']:
                logging.warning(
                    f"Resolution mismatch — device: {width}x{height}, "
                    f"config: {self.config['screen_width']}x{self.config['screen_height']}. "
                    "Auto-correcting."
                )
                self.config['screen_width'] = width
                self.config['screen_height'] = height
            else:
                logging.info(f"[OK] Screen resolution confirmed: {width}x{height}")
        except Exception as e:
            logging.warning(f"Could not verify screen resolution: {e}")
    
    def capture_screenshot(self) -> str:
        """Capture a screenshot from the device and return the local path."""
        timestamp = int(time.time())
        screenshot_path = str(
            Path(self.config['screenshot_dir']) /
            f"screen_{self.context['session_id']}_{timestamp}.png"
        )
        try:
            self.device.capture_screenshot(screenshot_path)
            logging.info(f"Screenshot captured: {screenshot_path}")
            self.context['screenshots'].append(screenshot_path)
            return screenshot_path
        except Exception as e:
            logging.error(f"Screenshot capture failed: {e}")
            raise
    
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on the device.
        
        Args:
            action: Action dictionary from Qwen3-VL
            
        Returns:
            Result dictionary with success status
        """
        try:
            action_type = action['action']
            logging.info(f"Executing: {action_type}")
            
            # Handle task completion
            if action_type == 'terminate':
                status = action.get('status', 'success')
                message = action.get('message', 'Task complete')
                logging.info(f"✓ Task {status}: {message}")
                return {
                    'success': True,
                    'action': action,
                    'task_complete': True
                }
            
            # Handle each action type
            if action_type == 'tap':
                self._execute_tap(action)
            
            elif action_type == 'swipe':
                self._execute_swipe(action)
            
            elif action_type == 'type':
                self._execute_type(action)
            
            elif action_type == 'wait':
                self._execute_wait(action)
            
            else:
                raise ValueError(f"Unknown action type: {action_type}")
            
            # Record action in history WITH coordinates
            action_record = {
                'action': action_type,
                'timestamp': time.time(),
                'elementName': action.get('observation', '')[:50]  # ← Observation from model
            }
            
            # Store coordinates for tap actions
            if action_type == 'tap' and 'coordinates' in action:
                action_record['coordinates'] = action['coordinates']  # ✓ Saved
            
            self.context['previous_actions'].append(action_record)
            
            # Standard delay after action
            time.sleep(self.config['step_delay'])
            
            return {
                'success': True,
                'action': action,
                'task_complete': False
            }
            
        except Exception as e:
            logging.error(f"Action execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'action': action,
                'task_complete': False
            }
    
    def _execute_tap(self, action: Dict[str, Any]):
        """Execute a tap action."""
        if 'coordinates' not in action:
            raise ValueError("Tap action missing coordinates")
        norm_x, norm_y = action['coordinates']
        x = max(0, min(int(norm_x * self.config['screen_width']),  self.config['screen_width']  - 1))
        y = max(0, min(int(norm_y * self.config['screen_height']), self.config['screen_height'] - 1))
        logging.info(f"Tapping at ({x}, {y}) [normalized: ({norm_x:.3f}, {norm_y:.3f})]")
        self.device.tap(x, y)

    def _execute_swipe(self, action: Dict[str, Any]):
        """Execute a swipe action."""
        direction = action.get('direction', 'up')
        cx = self.config['screen_width']  // 2
        cy = self.config['screen_height'] // 2
        d  = 0.7  # 70% of screen dimension

        if direction == 'up':
            x1, y1, x2, y2 = cx, cy, cx, int(cy * (1 - d))
        elif direction == 'down':
            x1, y1, x2, y2 = cx, cy, cx, int(cy * (1 + d))
        elif direction == 'left':
            x1, y1, x2, y2 = cx, cy, int(cx * (1 - d)), cy
        elif direction == 'right':
            x1, y1, x2, y2 = cx, cy, int(cx * (1 + d)), cy
        else:
            raise ValueError(f"Invalid swipe direction: {direction}")

        logging.info(f"Swiping {direction}: ({x1}, {y1}) -> ({x2}, {y2})")
        self.device.swipe(x1, y1, x2, y2)

    def _execute_type(self, action: Dict[str, Any]):
        """Execute a type action."""
        if 'text' not in action:
            raise ValueError("Type action missing text")
        recent = self.context['previous_actions'][-3:]
        if not any(a.get('action') == 'tap' for a in recent):
            logging.warning("Type action without recent tap — may fail")
        logging.info(f"Typing: {action['text']}")
        self.device.type_text(action['text'])

    def _execute_wait(self, action: Dict[str, Any]):
        """Execute a wait action."""
        wait_time = action.get('waitTime', 1000) / 1000.0
        logging.info(f"Waiting {wait_time:.1f}s...")
        time.sleep(wait_time)
    
    def _detect_repetitive_behavior(self) -> Optional[str]:
        """
        Detect if the agent is repeating the same failed actions.
        
        Returns:
            Warning message if repetitive behavior detected, None otherwise
        """
        if len(self.context['previous_actions']) < 3:
            return None
        
        # Get last N actions
        recent_actions = self.context['previous_actions'][-5:]
        
        # Count action types
        action_types = [a.get('action') for a in recent_actions]
        action_counter = Counter(action_types)
        
        # Check for repeated tap locations (within 5% tolerance)
        tap_actions = [a for a in recent_actions if a.get('action') == 'tap' and 'coordinates' in a]
        
        if len(tap_actions) >= 3:
            # Check if coordinates are similar (within 5% of screen)
            tolerance = 0.05
            # Group similar taps
            tap_groups = []
            for tap in tap_actions:
                coord = tap['coordinates']
                found_group = False
                for group in tap_groups:
                    if abs(coord[0] - group[0][0]) < tolerance and abs(coord[1] - group[0][1]) < tolerance:
                        group.append(coord)
                        found_group = True
                        break
                if not found_group:
                    tap_groups.append([coord])
            
            # Check if any group has threshold or more taps at SAME location
            for group in tap_groups:
                if len(group) >= self.config.get('loop_detection_threshold', 3):
                    return (
                        f"CRITICAL WARNING: You have tapped the same location {len(group)} times without success! "
                        f"This approach is NOT WORKING. You MUST try something completely different:\n"
                        f"1. Try swiping instead of tapping\n"
                        f"2. Go to home screen and start over\n"
                        f"3. Use search functionality\n"
                        f"4. Tap a DIFFERENT element\n"
                        f"DO NOT tap the same location again!"
                    )
        
        # Only warn about repeated action types if screen hasn't changed
        # This indicates the actions are having no effect
        if self.context.get('screen_unchanged', False):
            # Check if same action repeated in last 3 actions while screen unchanged
            last_3_actions = action_types[-3:] if len(action_types) >= 3 else action_types
            if len(set(last_3_actions)) == 1:  # All same action type
                action_type = last_3_actions[0]
                return (
                    f"CRITICAL WARNING: You've performed '{action_type}' {len(last_3_actions)} times "
                    f"and the screen hasn't changed! "
                    f"Your actions are having NO EFFECT. Try a COMPLETELY DIFFERENT approach immediately!"
                )
        
        # REMOVED: Excessive repetition warning (was causing false positives)
        # The above conditions are sufficient to detect actual loops
        
        return None
    
    def _get_enhanced_context(self, user_request: str) -> Dict[str, Any]:
        """
        Build enhanced context with failure information for the model.
        
        Args:
            user_request: The user's task request
            
        Returns:
            Enhanced context dictionary
        """
        enhanced_context = self.context.copy()
        
        # Add repetition warning if detected
        repetition_warning = self._detect_repetitive_behavior()
        if repetition_warning:
            enhanced_context['repetition_warning'] = repetition_warning
            logging.warning(repetition_warning)
        
        # Add failed attempts summary
        if self.context['failed_attempts']:
            recent_failures = self.context['failed_attempts'][-3:]  # Last 3 failures
            enhanced_context['recent_failures'] = [
                {
                    'action': f.get('action', 'unknown'),
                    'reason': f.get('reason', 'unknown'),
                    'timestamp': f.get('timestamp', 0)
                }
                for f in recent_failures
            ]
        
        return enhanced_context
    
    def execute_cycle(self, user_request: str) -> Dict[str, Any]:
        """
        Execute a single interaction cycle.
        
        Args:
            user_request: The user's task request
            
        Returns:
            Result dictionary
        """
        try:
            # Capture screenshot
            screenshot_path = self.capture_screenshot()
            
            # Calculate simple hash of screenshot to detect no-change scenarios
            import hashlib
            with open(screenshot_path, 'rb') as f:
                current_hash = hashlib.md5(f.read()).hexdigest()
            
            # Detect if screen hasn't changed (stuck)
            if self.context['last_screen_hash'] == current_hash:
                if len(self.context['previous_actions']) > 0:
                    logging.warning("Screen unchanged from last action - possible stuck state!")
                    # Add to context for model awareness
                    self.context['screen_unchanged'] = True
            else:
                self.context['screen_unchanged'] = False
            
            self.context['last_screen_hash'] = current_hash
            
            # Get enhanced context with failure tracking
            enhanced_context = self._get_enhanced_context(user_request)
            
            # Analyze with Qwen3-VL
            action = self.vl_agent.analyze_screenshot(
                screenshot_path,
                user_request,
                enhanced_context  # Pass enhanced context instead of basic context
            )
            
            if not action:
                raise Exception("Failed to get action from model")
            
            # Log model's observation and reasoning
            if 'observation' in action:
                logging.info(f"Model observation: {action['observation']}")
            if 'reasoning' in action:
                logging.info(f"Model reasoning: {action['reasoning']}")
            
            # Execute the action
            result = self.execute_action(action)
            
            # Track failures
            if not result['success']:
                self.context['failed_attempts'].append({
                    'action': action,
                    'reason': result.get('error', 'Unknown'),
                    'timestamp': time.time(),
                    'screenshot': screenshot_path
                })
            
            return result
            
        except Exception as e:
            logging.error(f"Cycle execution failed: {e}")
            raise
    
    def execute_task(self, user_request: str, max_cycles: int = None) -> Dict[str, Any]:
        """
        Execute a complete task through multiple cycles.
        
        Args:
            user_request: The user's task description
            max_cycles: Maximum number of action cycles (overrides config if provided)
            
        Returns:
            Task result dictionary
        """
        # Use provided max_cycles or fall back to config
        if max_cycles is None:
            max_cycles = self.config.get('max_cycles', 15)
        
        self.context['task_request'] = user_request
        logging.info("=" * 60)
        logging.info(f"STARTING TASK: {user_request}")
        logging.info(f"Max cycles: {max_cycles}")
        logging.info("=" * 60)
        
        cycles = 0
        task_complete = False
        last_error = None
        consecutive_failures = 0  # Track consecutive failures
        
        while cycles < max_cycles and not task_complete:
            cycles += 1
            logging.info(f"\n--- Cycle {cycles}/{max_cycles} ---")
            
            try:
                result = self.execute_cycle(user_request)
                
                if result.get('task_complete'):
                    task_complete = True
                    logging.info("[OK] Task marked complete by agent")
                    break
                
                if not result['success']:
                    consecutive_failures += 1
                    last_error = result.get('error', 'Unknown error')
                    logging.warning(f"Action failed: {last_error}")
                    logging.warning(f"Consecutive failures: {consecutive_failures}/{self.config['max_retries']}")
                    
                    # Stop if too many consecutive failures
                    if consecutive_failures >= self.config['max_retries']:
                        logging.error(f"Max consecutive retries ({self.config['max_retries']}) exceeded")
                        break
                else:
                    # Reset consecutive failures on success
                    consecutive_failures = 0
                
            except KeyboardInterrupt:
                logging.info("Task interrupted by user")
                raise
            except Exception as e:
                consecutive_failures += 1
                last_error = str(e)
                logging.error(f"Cycle error: {e}")
                
                if consecutive_failures >= self.config['max_retries']:
                    logging.error(f"Max consecutive retries ({self.config['max_retries']}) exceeded")
                    break
                
                # Wait before retry
                time.sleep(2)
        
        # Final verification if we hit max cycles
        if cycles >= max_cycles and not task_complete:
            logging.info("Max cycles reached, checking if task is actually complete...")
            screenshot_path = self.capture_screenshot()
            completion_check = self.vl_agent.check_task_completion(
                screenshot_path,
                user_request,
                self.context
            )
            
            if completion_check.get('complete'):
                task_complete = True
                logging.info(f"[OK] Task verified complete: {completion_check.get('reason')}")
        
        # Summary
        logging.info("\n" + "=" * 60)
        if task_complete:
            logging.info(f"[OK] TASK COMPLETED in {cycles} cycles")
            success = True
        else:
            logging.info(f"[FAIL] TASK INCOMPLETE after {cycles} cycles")
            if last_error:
                logging.info(f"Last error: {last_error}")
            success = False
        logging.info("=" * 60)
        
        return {
            'success': success,
            'cycles': cycles,
            'task_complete': task_complete,
            'context': self.context,
            'screenshots': self.context['screenshots']
        }


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python phone_agent.py 'your task here'")
        sys.exit(1)
    
    task = ' '.join(sys.argv[1:])
    
    # Load config
    config_path = 'config.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Run task
    agent = PhoneAgent(config)
    result = agent.execute_task(task)
    
    if result['success']:
        print(f"\n✓ Task completed in {result['cycles']} cycles")
    else:
        print(f"\n✗ Task failed after {result['cycles']} cycles")