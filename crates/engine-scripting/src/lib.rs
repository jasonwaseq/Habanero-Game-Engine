//! Rhai-based scripting host.

use anyhow::Result;
use engine_ecs::EventBus;
use rhai::{Engine, Scope};

pub struct ScriptHost {
    engine: Engine,
}

impl ScriptHost {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        engine.register_fn("log_info", |msg: &str| tracing::info!("{msg}"));
        Self { engine }
    }

    pub fn run_script(
        &self,
        source: &str,
        scope: &mut Scope<'_>,
        event_bus: &EventBus,
    ) -> Result<()> {
        self.engine
            .eval_with_scope::<()>(scope, source)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        event_bus.push(String::from("script_executed"));
        Ok(())
    }
}

impl Default for ScriptHost {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_script_and_emits_event() {
        let host = ScriptHost::new();
        let bus = EventBus::default();
        let mut scope = Scope::new();
        host.run_script("let x = 1 + 2;", &mut scope, &bus)
            .expect("script runs");
        let events = bus.drain::<String>();
        assert_eq!(events, vec!["script_executed".to_string()]);
    }

    #[test]
    fn invalid_script_returns_error() {
        let host = ScriptHost::new();
        let bus = EventBus::default();
        let mut scope = Scope::new();
        let result = host.run_script("let x = ;", &mut scope, &bus);
        assert!(result.is_err());
    }

    #[test]
    fn host_exposes_log_info_binding() {
        let host = ScriptHost::new();
        let bus = EventBus::default();
        let mut scope = Scope::new();
        host.run_script(r#"log_info("hello from rhai");"#, &mut scope, &bus)
            .expect("log_info available");
    }
}
