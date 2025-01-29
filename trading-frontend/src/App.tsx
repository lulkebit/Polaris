import { AnalysisDashboard } from './components/AnalysisDashboard';

function App() {
    return (
        <div className='min-h-screen bg-gray-50 w-full'>
            <nav className='bg-white shadow-sm w-full'>
                <div className='w-full max-w-[95%] mx-auto px-4 py-4'>
                    <h1 className='text-2xl font-bold text-gray-800'>
                        Trading AI Analysis
                    </h1>
                </div>
            </nav>
            <main className='w-full'>
                <AnalysisDashboard />
            </main>
        </div>
    );
}

export default App;
