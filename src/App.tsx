import { SpeedInsights } from '@vercel/speed-insights/react';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Genesis Conductor</h1>
        <p>AI-native application scaffolding engine</p>
        <p>The official open-source implementation of the "Master Control Program"</p>
      </header>
      <SpeedInsights />
    </div>
  );
}

export default App;
