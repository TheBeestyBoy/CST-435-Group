import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import Header from './components/Header'
import LoadingSpinner from './components/LoadingSpinner'

// Lazy load pages for better performance
const Home = lazy(() => import('./pages/Home'))
const ANNProject = lazy(() => import('./pages/ANNProject'))
const CNNProject = lazy(() => import('./pages/CNNProject'))
const NLPProject = lazy(() => import('./pages/NLPProject'))
const RNNProject = lazy(() => import('./pages/RNNProject'))
const GAProject = lazy(() => import('./pages/GAProject'))
const GANProject = lazy(() => import('./pages/GANProject'))
const GANModelDetails = lazy(() => import('./pages/GANModelDetails'))
const RLProject = lazy(() => import('./pages/RLProject'))
const Resume = lazy(() => import('./pages/Resume'))

function App() {
  return (
    <Router>
      <div className="min-h-screen">
        <Header />
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/ann" element={<ANNProject />} />
            <Route path="/cnn" element={<CNNProject />} />
            <Route path="/nlp" element={<NLPProject />} />
            <Route path="/rnn" element={<RNNProject />} />
            <Route path="/ga" element={<GAProject />} />
            <Route path="/gan" element={<GANProject />} />
            <Route path="/gan/model" element={<GANModelDetails />} />
            <Route path="/rl" element={<RLProject />} />
            <Route path="/resume" element={<Resume />} />
          </Routes>
        </Suspense>
      </div>
    </Router>
  )
}

export default App
