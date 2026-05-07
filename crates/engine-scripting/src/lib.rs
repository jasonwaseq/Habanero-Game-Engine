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
